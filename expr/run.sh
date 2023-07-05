cd ../

# model_size="13b"
model_size="7b"
WANDB_DISABLED=true

# NOTE by zian: QLoRA still only uses one GPU for computation
# CUDA_VISIBLE_DEVICES=4,5 python src/train_dcrn.py \
#     --model_name_or_path "huggyllama/llama-${model_size}" \
#     --do_train \
#     --dataset gh-dataset-train.jsonl \
#     --dataset_dir data/dcrn \
#     --max_source_length 1700 \
#     --max_target_length 200 \
#     --prompt_template dcrn-end2end \
#     --dev_ratio 0.05 \
#     --preprocessing_num_workers 4 \
#     --finetuning_type lora \
#     --output_dir "save/llama-${model_size}-qlora-let" \
#     --quantization_bit 4 \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --save_total_limit 10 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --plot_loss \
#     --fp16

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=12345 src/train_dcrn.py \
    --model_name_or_path "huggyllama/llama-${model_size}" \
    --do_train \
    --do_eval \
    --dataset gh-dataset-sample.jsonl \
    --dataset_dir data/dcrn \
    --max_source_length 1700 \
    --max_target_length 200 \
    --prompt_template dcrn-end2end \
    --dev_ratio 0.01 \
    --preprocessing_num_workers 8 \
    --overwrite_cache \
    --finetuning_type lora \
    --output_dir "save/llama-${model_size}-lora-full" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16