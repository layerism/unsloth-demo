#!/bin/bash

CUDA_VISIBLE_DEVICES=7 \
VLLM_LOGGING_LEVEL=DEBUG \
python -m vllm.entrypoints.openai.api_server \
    --model $1 \
    --host 0.0.0.0 \
    --port $3 \
    --max-model-len 8192 \
    --enable-lora \
    --lora-modules joke_qwen3=$2 \
    --served-model-name joke_qwen3 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 \
    --trust-remote-code \
    --chat-template "./tests/qwen3.jinja" \
    --enable-log-requests \
    --enable-log-outputs