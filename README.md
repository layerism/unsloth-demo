# 适用于本科生实训的 llm fintune 以及 vllm 部署教程
1. 本课程使用 unsloth 作为 llm fintune 的工具，使用 vllm 作为大模型部署的工具。

## 环境依赖
```shell
conda create -n unsloth python=3.11
conda activate unsloth
pip install unsloth==2025.12.9

conda create -n vllm python=3.11
conda activate vllm
pip install vllm==0.11.2
```

## 安装