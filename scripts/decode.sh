#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

FIELD=restaurant
MODEL_SAVE_PATH=baseline_${FIELD}

python generate.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_SAVE_PATH} \
    --num_samples 5 \
    --input_file=data/${FIELD}/test.txt \
    --top_k 5 \
    --output_file=${MODEL_SAVE_PATH}/results.json \
    --length 80
