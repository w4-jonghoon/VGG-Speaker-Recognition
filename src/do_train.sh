#!/bin/bash

python train_more_tts.py \
    --gpu 0,1 \
    --resume ../model/weights.h5 \
    --batch_size 160 \
    --net resnet34s \
    --lr 0.001 \
    --warmup_ratio 0.1 \
    --optimizer adam \
    --epochs 1 \
    --multiprocess 1 \
    --loss softmax \
    --data_path /nas0/poodle/speech_dataset/uncategorized/more-TTS-Jun2019

