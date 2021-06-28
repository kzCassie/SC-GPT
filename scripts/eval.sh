#!/bin/bash

FIELD=taxi
MODEL_SAVE_PATH=baseline_${FIELD}

python evaluator.py \
    --domain ${FIELD} \
    --target_file ${MODEL_SAVE_PATH}/results.json | tee ${MODEL_SAVE_PATH}/eval.txt