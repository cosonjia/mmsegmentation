from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

# config_file =  '../configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py'
# checkpoint_file = '../checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth'
# config_file = '../configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
# checkpoint_file = '../checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# config_file = '../configs/pspnet/pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k.py'
# checkpoint_file = '../checkpoints/pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k_20210820_203258-b88df27f.pth'

config_file = '../configs/pspnet/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k.py'
checkpoint_file = '../checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# test a single image\n
img = 'img.png'
result = inference_segmentor(model, img)
# show the results\n
show_result_pyplot(model, img, result, get_palette('coco-stuff'))
# show_result_pyplot(model, img, result, get_palette('voc12aug'))
# show_result_pyplot(model, img, result, get_palette('cityscapes'))