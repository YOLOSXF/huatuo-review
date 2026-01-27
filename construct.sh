python construct_verifiable_medical_problems_ch.py \
    --data_path ./data/train_zh.jsonl \
    --api_key "sk-bae4a942ee3541cf8c8bf2d8351ca206" \
    --model_name "qwen-max" \
    --api_url "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" \
    --limit_num 100 