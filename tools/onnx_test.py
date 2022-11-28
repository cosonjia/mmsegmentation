# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
import tempfile
import mmcv
import numpy as np
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.runner import (get_dist_info, init_dist)

from mmseg import digit_version
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
import cv2

from mmseg.datasets.voc import PascalVOCDataset


def show_result(img,
                result,
                palette,
                classes,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result list/tuple: The each element is ndarray or Tensor. The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    seg = result[0]
    if palette is None:
        if palette is None:
            # Get random state before set seed,
            # and restore random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(
                0, 255, size=(len(classes), 3))
            np.random.set_state(state)
        else:
            palette = palette
    palette = np.array(palette)
    assert palette.shape[0] == len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    if not (show or out_file):
        warnings.warn('show==False and out_file is not specified, only '
                      'result image will be returned')
        return img


class PSPNetPostProcessor(object):
    def processing(self, ort_outs, img_metas, **kwargs):
        batch_size = ort_outs[0].shape[0]
        outs = []
        for i in range(batch_size):
            ori_shape = img_metas[i]['ori_shape']
            # print('ori_shape: ', ori_shape)
            ort_res = cv2.resize(ort_outs[0][i][0].astype(np.uint8), (ori_shape[1], ori_shape[0]))
            outs.append(ort_res)
        return outs


class ONNXModel(object):
    def __init__(self, onnx_file, class_names=None, device_id=0, postprocessor=None):
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        sess = ort.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'], sess_options=session_options)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})

        sess.set_providers(providers, options)

        self.sess = sess
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        self.input_name = sess.get_inputs()[0].name
        self.is_cuda_available = is_cuda_available
        self.postprocessor = postprocessor
        self.CLASSES = class_names
        self.device_id = device_id

    def processing(self, img, img_metas, **kwargs):
        outputs = self.__run_onnx(img, img_metas, **kwargs)
        if isinstance(img_metas[0].data, dict):
            img_metas = [img_metas[0].data]
        else:
            img_metas = img_metas[0].data[0]
        if self.postprocessor is not None:
            bbox_results = self.postprocessor.processing(outputs,
                                                         img_metas=img_metas,
                                                         **kwargs)
        else:
            bbox_results = None
        return outputs, bbox_results

    def __run_onnx2(self, imgs, img_metas, **kwargs):
        input_data = imgs[0]
        if isinstance(input_data, (list, tuple)):
            input_data = input_data[0]
        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'
        if not self.is_cuda_available and torch is not None and isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu()
        if torch is not None and isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device_type)
            buffer_ptr = input_data.data_ptr()
        else:
            buffer_ptr = input_data.ctypes.data
        self.io_binding.bind_input(
            name=self.input_name,  # 'input',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=input_data.shape,
            buffer_ptr=buffer_ptr)

        for name in self.output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        ort_outputs = self.io_binding.copy_outputs_to_cpu()
        return ort_outputs

    def __run_onnx(self, imgs, img_metas, **kwargs):
        input_data = imgs[0]
        if isinstance(input_data, (list, tuple)):
            input_data = input_data[0]
        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'
        if not self.is_cuda_available and torch is not None and isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu()
        if torch is not None and isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device_type)
            input = input_data.numpy()
        else:
            input = input_data
        ort_outputs = self.sess.run([], {self.input_name: input})
        return ort_outputs


def build_onnx_model(cfg, test_cfg=None):
    model = ONNXModel(cfg.model, postprocessor=PSPNetPostProcessor())
    return model


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            _, result = model.processing(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                img_show=img_meta['filename']
                show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    classes=dataset.CLASSES,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', help='The path of config file')
    parser.add_argument('--model', help='The path of model file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')

    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_onnx_model(args, test_cfg=cfg.get('test_cfg'))
    CLASSES = dataset.CLASSES
    PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {}

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')
    eval_on_format_results = (
            args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    cfg.device = get_device()
    distributed = False
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        # model = revert_sync_batchnorm(model)
        # model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
