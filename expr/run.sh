cd ../

# model_size="13b"
model_size="7b"
WANDB_DISABLED=true
# nohup python src/train_dcrn.py \
#     --model_name_or_path "huggyllama/llama-${model_size}" \
#     --do_train \
#     --dataset alpaca_en \
#     --finetuning_type lora \
#     --output_dir "output/llama-${model_size}-qlora-let" \
#     --quantization_bit 4 \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --plot_loss \
#     --fp16 > "llama-${model_size}-qlora-let.log" 2>&1

CUDA_VISIBLE_DEVICES=5 python src/train_dcrn.py \
    --model_name_or_path "huggyllama/llama-${model_size}" \
    --do_train \
    --dataset gh-dataset-sample.jsonl \
    --dataset_dir data/dcrn \
    --finetuning_type lora \
    --output_dir "save/llama-${model_size}-qlora-let" \
    --quantization_bit 4 \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16