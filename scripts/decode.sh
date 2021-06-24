#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
MODEL_SAVE_PATH=saved_models/baseline

python generate.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_SAVE_PATH} \
    --num_samples 5 \
    --input_file=data/restaurant/test.txt \
    --top_k 5 \
    --output_file=${MODEL_SAVE_PATH}/results.json \
    --length 80
