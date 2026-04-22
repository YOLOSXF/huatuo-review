#!/bin/bash
# 启动 SGLang 服务 + 执行评估的一体化脚本

# ===================== 核心配置参数（可修改）=====================
log_num=2
model_name="/jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/ckpts/sft_stage1/checkpoint-1-210/tfmr"  # 模型路径
cuda_device=2                                    # GPU 设备编号（多卡用 "0,1"）
mem_fraction_static=0.8                          # 静态内存占比
dp=1                                             # 数据并行数
tp=1                                             # 张量并行数
eval_file="/jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/evaluation/data/eval_filtered.json"        # 评估数据文件路径
eval_delay=50                                  # 启动服务后等待时间（秒），确保服务就绪
# ===================== 日志目录（无需修改） =====================
logs_dir="/jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/evaluation/logs"
# ====================================================================

# 计算端口号（保持原逻辑）
port="28${log_num}35"
# 服务日志文件
server_log="sglang${log_num}.log"
# 评估日志文件（单独保存，便于区分）
eval_log="eval${log_num}.log"

# ===================== 工具函数：检查进程是否存在 =====================
check_process() {
  local pid=$1
  ps -p $pid > /dev/null 2>&1
}

# ===================== 步骤1：启动 SGLang 服务 =====================
echo "========================================"
echo "          1. 启动 SGLang 服务           "
echo "========================================"
echo "模型路径: $model_name"
echo "GPU 设备: $cuda_device"
echo "端口号: $port"
echo "服务日志: $server_log"
echo "内存占比: $mem_fraction_static"
echo "DP/TP: $dp/$tp"
echo "========================================"

# 启动服务（后台运行，记录进程ID）
CUDA_VISIBLE_DEVICES=$cuda_device python -m sglang.launch_server \
  --model-path "$model_name" \
  --port "$port" \
  --mem-fraction-static "$mem_fraction_static" \
  --dp "$dp" \
  --tp "$tp" \
  > "$logs_dir/$server_log" 2>&1 &

# 获取服务进程ID
server_pid=$!
echo "服务启动中，进程ID: $server_pid"
echo "等待 $eval_delay 秒确保服务就绪..."
sleep $eval_delay

# 检查服务是否启动成功
if check_process $server_pid && curl -fsS "http://127.0.0.1:${port}/v1/models" > /dev/null; then
  echo "✅ 服务启动成功！"
else
  echo "❌ 服务未就绪（进程可能退出或接口不可用）"
  echo "查看服务日志：tail -n 200 $logs_dir/$server_log"
  exit 1
fi



# ===================== 步骤2：执行评估 =====================
echo -e "\n========================================"
echo "              2. 执行模型评估            "
echo "========================================"
echo "评估文件: $eval_file"
echo "评估日志: $eval_log"
echo "连接端口: $port"
echo "========================================"

# 仅评估重试参数
eval_max_retries=3
eval_retry_interval=10

# 清空旧日志（避免多次运行混在一起）
: > "$logs_dir/$eval_log"

eval_ok=0
for attempt in $(seq 1 $eval_max_retries); do
  echo "[Eval] 尝试 ${attempt}/${eval_max_retries} ..." | tee -a "$logs_dir/$eval_log"

  # 执行评估命令（日志追加保存）
  python evaluation/eval.py \
    --model_name "$model_name" \
    --eval_file "$eval_file" \
    --port "$port" \
    --strict_promp \
    --task "tuning1_other" \
    >> "$logs_dir/$eval_log" 2>&1

  if [ $? -eq 0 ]; then
    eval_ok=1
    echo "✅ 评估执行完成！" | tee -a "$logs_dir/$eval_log"
    echo "查看评估日志：tail -f $logs_dir/$eval_log"
    break
  else
    echo "❌ 评估执行失败（尝试 ${attempt}/${eval_max_retries}）" | tee -a "$logs_dir/$eval_log"
    echo "查看错误日志：tail -n 200 $logs_dir/$eval_log"
    if [ $attempt -lt $eval_max_retries ]; then
      echo "[Eval] 等待 ${eval_retry_interval}s 后重试..." | tee -a "$logs_dir/$eval_log"
      sleep $eval_retry_interval
    fi
  fi
done

if [ $eval_ok -ne 1 ]; then
  echo "❌ 评估重试仍失败，退出。"
  echo "服务仍在运行（PID: $server_pid），如需停止：kill -9 $server_pid"
  exit 1
fi

echo -e "\n========================================"
echo "          流程结束！                     "
echo "========================================"