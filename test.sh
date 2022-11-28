#!/bin/bash

python3 tools/test.py \
    configs/pspnet/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k.py \
    checkpoints/pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pth \
    --work-dir  test_space \
    --out  pspnet_r50-d8_512x512_4x4_80k_coco-stuff164k_20210707_152034-0e41b2db.pkl  \
    --eval   mIoU  mDice mFscore



python3 tools/test.py \
    configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py \
    checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth \
    --work-dir  test_space \
    --out  pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pkl  \
    --eval  mIoU  mDice mFscore

python3 tools/test.py \
    configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py \
    checkpoints/pspnet_r50-d8_512x512_20k_voc12aug_512x512.onnx \
    --work-dir  test_space \
    --out  pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pkl  \
    --eval  mIoU  mDice mFscore

# ADE20K
python3 tools/test.py \
    configs/pspnet/pspnet_r50-d8_512x512_80k_ade20k.py \
    checkpoints/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth \
    --work-dir  test_space \
    --out pspnet_r50-d8_512x512_80k_ade20k_512x512.pkl  \
    --eval  mIoU  mDice mFscore

python3 tools/test.py \
    configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py \
    checkpoints/pspnet_r50-d8_512x512_160k_ade20k_512x512.onnx \
    --work-dir  test_space \
    --out pspnet_r50-d8_512x512_160k_ade20k_512x512.pkl  \
    --eval  mIoU  mDice mFscore
python3 tools/test.py \
    configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k.py \
    checkpoints/pspnet_r101-d8_512x512_80k_ade20k_512x512.onnx \
    --work-dir  test_space \
    --out pspnet_r101-d8_512x512_80k_ade20k_512x512.pkl  \
    --eval  mIoU  mDice mFscore
python3 tools/test.py \
    configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py \
    checkpoints/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth \
    --work-dir  test_space \
    --out pspnet_r101-d8_512x512_160k_ade20k_512x512.pkl  \
    --eval  mIoU  mDice mFscore