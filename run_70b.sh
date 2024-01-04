seq_lens=(4096 2048 1024 512 256)
max_batch=(4 8 16 32 64)
# batches=(1 2 4 8)

# 使用 zip 结合两个数组
for i in "${!seq_lens[@]}"; do
    seq_len=${seq_lens[$i]}
    max_batch=${max_batch[$i]}
    for batch in $(seq 1 1 $max_batch);
    # for j in "${!batches[@]}";
    do
            # batch=${batches[$j]}
            echo "[seq_len $seq_len, batch $batch]"
            accelerate launch --config_file /home/siqizhu/deepspeed_peft/default_config.yaml /home/siqizhu/deepspeed_peft/deepspeed_baseline.py --seq_len=$seq_len --batch=$batch --model_name_or_path="/mnt/data/zhongrx/Llama-2-70b-hf"
    done
done
