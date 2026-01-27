python search_for_complex_reasoning_path_chinese_version.py \
    --data_path  /jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/train_zh_final_100.json \
    --efficient_search True  \
    --max_search_attempts 1 \
    --max_search_depth 2 \
    --api_key "sk-bae4a942ee3541cf8c8bf2d8351ca206" \
    --model_name "kimi-k2-thinking" \
    --api_url "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" \
    --limit_num 10