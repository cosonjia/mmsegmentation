from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot, inference
from mmseg.core.evaluation import get_palette
import mmcv
from mmseg.datasets.pipelines import Compose
from mmseg.apis import init_segmentor, inference_segmentor
import cv2
import onnxruntime as ort
import torch
import numpy as np


def show_result(img,
                result,
                CLASSES,
                palette=None,
                win_name='',
                show=True,
                wait_time=0,
                out_file=None,
                opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
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
        if PALETTE is None:
            # Get random state before set seed,
            # and restore random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(
                0, 255, size=(len(CLASSES), 3))
            np.random.set_state(state)
        else:
            palette = PALETTE
    palette = np.array(palette)
    assert palette.shape[0] == len(CLASSES)
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
        import warnings
        warnings.warn('show==False and out_file is not specified, only '
                      'result image will be returned')
        return img


# config_file =  '../configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py'
# checkpoint_file = '../checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth'
# config_file = '../configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
# checkpoint_file = '../checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# config_file = '../configs/pspnet/pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k.py'
# checkpoint_file = '../checkpoints/pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k_20210820_203258-b88df27f.pth'

config_file = '../configs/pspnet/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k.py'
checkpoint_file = '../checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth'
onnx_file = '../checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_dynamic.onnx'
config = mmcv.Config.fromfile(config_file)
# build the model from a config file and a checkpoint file
# test_pipeline = Compose(config.data.test.pipeline)
test_pipeline = Compose([inference.LoadImage()] + config.data.test.pipeline[1:])
session_options = ort.SessionOptions()
sess = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'], sess_options=session_options)
modelmeta = sess.get_modelmeta()
providers = ['CPUExecutionProvider']
options = [{}]
is_cuda_available = ort.get_device() == 'GPU'
if is_cuda_available:
    providers.insert(0, 'CUDAExecutionProvider')
    options.insert(0, {'device_id': 0})
sess.set_providers(providers, options)
io_binding = sess.io_binding()
output_names = [_.name for _ in sess.get_outputs()]
input_name = sess.get_inputs()[0].name
## coco-stuff
CLASS = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush',
    'cabinet',
    'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard',
    'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile',
    'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill',
    'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
    'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
    'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
    'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
    'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
    'window-blind', 'window-other', 'wood')
PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224], [0, 192, 192],
           [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64], [0, 160, 192],
           [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192], [128, 128, 0],
           [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
           [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32], [128, 96, 128],
           [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128], [128, 128, 128],
           [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192], [0, 128, 32],
           [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
           [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192], [0, 0, 32], [64, 160, 128], [128, 64, 64],
           [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128], [128, 192, 0], [128, 0, 96],
           [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224], [64, 32, 0],
           [0, 192, 128], [64, 128, 224], [192, 160, 0], [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
           [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224], [64, 96, 128], [128, 192, 128], [64, 0, 224],
           [192, 224, 128], [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192], [0, 128, 96], [0, 224, 0],
           [64, 64, 64], [128, 128, 224], [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0], [64, 192, 64],
           [128, 128, 96], [128, 32, 128], [64, 0, 192], [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
           [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128], [192, 128, 0], [128, 64, 32], [128, 32, 64],
           [192, 0, 128], [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160], [0, 32, 64], [64, 128, 128],
           [64, 192, 160], [128, 160, 64], [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128], [64, 64, 32],
           [0, 224, 192], [192, 0, 0], [192, 64, 160], [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
           [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192], [0, 192, 32], [64, 224, 64], [64, 0, 64],
           [128, 192, 160], [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64], [64, 128, 64], [128, 192, 32],
           [192, 32, 192], [64, 64, 192], [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160], [64, 32, 192],
           [192, 192, 192], [0, 64, 160], [192, 160, 192], [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
           [64, 192, 96], [64, 160, 64], [64, 64, 0]]

# test a single image\n
img = 'demo.png'
img_data = dict(img=img)
img_data = test_pipeline(img_data)
input_data = img_data['img'][0]
ori_shape = img_data['img_metas'][0].data['ori_shape']
if input_data.dim() == 3:
    input_data_local = input_data.unsqueeze(0)
else:
    input_data_local = input_data

input_data = np.load('../demo_process.npy')
np.testing.assert_allclose(input_data.astype(np.float32),
                           input_data_local.cpu().numpy().astype(np.float32),
                           rtol=1e-7, atol=1e-7,
                           err_msg="The preprocessing result are different between demo and local")
# input_data=input_data_local
if isinstance(input_data, torch.Tensor):
    # buffer_ptr = input_data.data_ptr()
    buffer_ptr = input_data.detach().numpy().ctypes.data
else:
    buffer_ptr = input_data.ctypes.data

io_binding.bind_input(
    name=input_name,  # 'input',
    device_type='cuda' if is_cuda_available else 'cpu',
    device_id=0,
    element_type=np.float32,
    shape=input_data.shape,
    buffer_ptr=buffer_ptr)
for name in output_names:
    io_binding.bind_output(name)
sess.run_with_iobinding(io_binding)
ort_outputs = io_binding.copy_outputs_to_cpu()
demo_ort_result = np.load('../demo_ort_result.npy').squeeze()
"""
# resize onnx_result to ori_shape
onnx_result = ort_outputs[0].squeeze()
np.testing.assert_allclose(demo_ort_result.astype(np.float32),
                           onnx_result.astype(np.float32),
                           rtol=1e-7, atol=1e-7,
                           err_msg="The inference result are different between demo and local in binding model")


onnx_result_r = sess.run(
            None, {'input': input_data_local.detach().numpy()})[0][0][0]
np.testing.assert_allclose(demo_ort_result.astype(np.float32),
                           onnx_result_r.astype(np.float32),
                           rtol=1e-7, atol=1e-7,
                           err_msg="The inference result are different between demo and local in running model")
onnx_result_ = cv2.resize(onnx_result[0].astype(np.uint8),
                          (ori_shape[1], ori_shape[0]))
print(ort_outputs)
"""
demo_ort_result = cv2.resize(demo_ort_result.astype(np.uint8),
                          (ori_shape[1], ori_shape[0]))
show_result(img, (demo_ort_result,), CLASS, PALETTE)

# show the results\n
# show_result_pyplot(model, img, result, get_palette('coco-stuff'))
# show_result_pyplot(model, img, result, get_palette('voc12aug'))
# show_result_pyplot(model, img, result, get_palette('cityscapes'))
