#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DOMAIN=restaurant
MODEL_SAVE_PATH=baseline_${DOMAIN}

#python generate.py \
#    --model_type=gpt2 \
#    --model_name_or_path=${MODEL_SAVE_PATH} \
#    --num_samples 5 \
#    --input_file=data/${DOMAIN}/test.txt \
#    --top_k 5 \
#    --output_file=${MODEL_SAVE_PATH}/results.json \
#    --length 80

# T5
MODEL_TYPE=t5
DOMAIN=restaurant
MODEL_SAVE_PATH=t5/no_task_pretrain

python generate.py \
    --model_type=${MODEL_TYPE} \
    --model_name_or_path=${MODEL_SAVE_PATH} \
    --num_samples 1 \
    --input_file=data/${DOMAIN}/test.txt \
    --top_k 5 \
    --output_file=${MODEL_SAVE_PATH}/results.json \
    --length 80