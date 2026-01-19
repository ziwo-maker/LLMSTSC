#!/bin/bash

# 设置环境变量
# export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2

# 设置工作目录（根据你的实际路径修改）
cd /home/code/LLMTSCS/


CUDA_VISIBLE_DEVICES=0,1,2,3 python run_open_LLM_with_vllm.py     --llm_model gemma-3-27B     --llm_path /home/data/model/gemma-3-27B/     --dataset jinan     --traffic_file anon_3_4_jinan_synthetic_24h_6000.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_open_LLM_with_vllm.py     --llm_model gemma-3-27B     --llm_path /home/data/model/gemma-3-27B/     --dataset newyork_28x7     --traffic_file anon_28_7_newyork_real_triple.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_random.py        --dataset jinan     --traffic_file anon_3_4_jinan_synthetic_24h_6000.json

# 定义通用参数
LLM_MODEL="gemma-3-27B"
LLM_PATH="/home/data/model/gemma-3-27B/"
PROJ_NAME="TSCS_jinanreal2000"

echo "开始运行实验..."
echo "================================"
·
# 运行 Hangzhou 数据集
echo "运行 Hangzhou 数据集..."
python run_open_LLM.py \
    --llm_model gemma-3-27B \
    --llm_path /home/data/model/gemma-3-27B/ \
    --dataset jinan \
    --traffic_file anon_3_4_jinan_real_2000.json \
    --proj_name "$PROJ_NAME"

if [ $? -eq 0 ]; then
    echo "Hangzhou 数据集完成 ✓"
else
    echo "Hangzhou 数据集失败 ✗"
fi

echo "================================"

# 运行 Jinan 数据集 - 2000
echo "运行 Jinan 数据集 (2000)..."
python run_open_LLM.py \
    --llm_model deepseek-32B \
    --llm_path /home/data/model/deepseek-32B/ \
    --dataset jinan \
    --traffic_file anon_3_4_jinan_real_2000.json \
    --proj_name "$PROJ_NAME"

if [ $? -eq 0 ]; then
    echo "Jinan 2000 数据集完成 ✓"
else
    echo "Jinan 2000 数据集失败 ✗"
fi

echo "================================"

# 运行 Jinan 数据集 - 2500
echo "运行 Jinan 数据集 (2500)..."
python run_open_LLM.py \
    --llm_model Mistral-13B \
    --llm_path /home/data/model/Mistral-13B/ \
    --dataset jinan \
    --traffic_file anon_3_4_jinan_real_2000.json \
    --proj_name "$PROJ_NAME"

if [ $? -eq 0 ]; then
    echo "Jinan 2500 数据集完成 ✓"
else
    echo "Jinan 2500 数据集失败 ✗"
fi
# with default methods of Transformers

                       
# or with VLLM (much faster but will cost more GPU memory)

echo "================================"

# 运行 Jinan 数据集 - 完整
echo "运行 Jinan 数据集 (完整)..."
python run_open_LLM.py \
    --llm_model Qwen-32B \
    --llm_path /home/data/model/Qwen-32B/ \
    --dataset jinan \
    --traffic_file anon_3_4_jinan_real_2000.json \
    --proj_name "$PROJ_NAME"

if [ $? -eq 0 ]; then
    echo "Jinan 完整数据集完成 ✓"
else
    echo "Jinan 完整数据集失败 ✗"
fi

echo "运行 Jinan 数据集 (完整)..."
python run_open_LLM.py \
    --llm_model Qwen3-VL-32B \
    --llm_path /home/data/model/Qwen3-VL-32B/ \
    --dataset jinan \
    --traffic_file anon_3_4_jinan_real_2000.json \
    --proj_name "$PROJ_NAME"

if [ $? -eq 0 ]; then
    echo "Jinan 完整数据集完成 ✓"
else
    echo "Jinan 完整数据集失败 ✗"
fi


echo "运行 Jinan 数据集 (完整)..."
python run_open_LLM.py \
    --llm_model Qwen3-VL-30B-A3B-Thinking \
    --llm_path /home/data/model/Qwen3-VL-30B-A3B-Thinking/ \
    --dataset jinan \
    --traffic_file anon_3_4_jinan_real_2000.json \
    --proj_name "$PROJ_NAME"

if [ $? -eq 0 ]; then
    echo "Jinan 完整数据集完成 ✓"
else
    echo "Jinan 完整数据集失败 ✗"
fi
