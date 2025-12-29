#!/bin/bash

VLLM_USE_V1=0 vllm serve $1 \
    --host 0.0.0.0 \
    --port $3 \
    --enable-lora \
    --lora-modules unsloth-model=$2 \
    --served-model-name unsloth-model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 \
    --trust-remote-code \
    --chat-template "./tests/qwen3.jinja" \
    --enable-log-requests \
    --enable-log-outputs