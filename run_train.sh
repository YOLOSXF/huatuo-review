export CUDA_VISIBLE_DEVICES=1,2

ACCELERATE_LOG_LEVEL=info \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 2  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard SFT_stage1.py \
    --model_path /jeyzhang/shenxiaofeng/Liver/models_cache/qwen2.5-7b-instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
    --data_path /jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/data/medical_o1_sft.json \
    --output_dir ./ckpts \
    --log_dir ./train_logs 