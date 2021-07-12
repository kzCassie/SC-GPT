#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DOMAIN=taxi
MODEL_SAVE_PATH=baseline_${DOMAIN}
PRE_TRINED_MODEL_PATH=scgpt/
#gpt2, gpt2-meidum
EPOCH=5
LR=5e-5

python train.py \
    --seed 42 \
    --output_dir=${MODEL_SAVE_PATH} \
    --model_type=gpt2 \
    --model_name_or_path=${PRE_TRINED_MODEL_PATH} \
    --do_train \
    --do_eval \
    --eval_data_file=data/${DOMAIN}/train.txt \
    --per_gpu_train_batch_size 1 \
    --num_train_epochs ${EPOCH} \
    --learning_rate ${LR} \
    --overwrite_cache \
    --use_tokenize \
    --train_data_file=data/${DOMAIN}/train.txt \
    --overwrite_output_dir
