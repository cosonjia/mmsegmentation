#!/bin/bash
# kNET
python3 tools/pytorch2onnx.py \
configs/knet/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k.py  --checkpoint checkpoints/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_043751-abcab920.pth \
--input-img demo/demo.png  \
--shape 512 512 \
--output-file checkpoints/fcheckpoints/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_043751-abcab920.onnx \
--show --verify

# PsPNet
##
python3 tools/pytorch2onnx.py \
configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py  --checkpoint checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth \
--input-img demo/demo.png  \
--shape 512 512 \
--output-file checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.onnx \
--show --verify
##pspnet_r50-d8_512x1024_40k_cityscapes
python3 tools/pytorch2onnx.py \
configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py  --checkpoint checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth \
--input-img demo/demo.png  \
--shape 512 1024 \
--output-file checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.onnx \
--show --verify

## pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k
python3 tools/pytorch2onnx.py \
configs/pspnet/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k.py  --checkpoint checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth \
--input-img demo/demo.png  \
--shape 512 512 \
--output-file checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_512x512.onnx \
--show --verify
# dynamic-export
python3 tools/pytorch2onnx.py \
configs/pspnet/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k.py  --checkpoint checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth \
--input-img demo/demo.png  \
--shape 512 512 \
--output-file checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_dynamic.onnx \
--show --verify --dynamic-export