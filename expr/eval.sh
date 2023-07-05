cd ../

# model_size="13b"
model_size="7b"
WANDB_DISABLED=true

CUDA_VISIBLE_DEVICES=4 python src/train_dcrn.py \
    --model_name_or_path "huggyllama/llama-${model_size}" \
    --do_predict \
    --predict_with_generate \
    --dataset gh-dataset-test.jsonl \
    --dataset_dir data/dcrn \
    --max_source_length 1700 \
    --max_target_length 200 \
    --prompt_template dcrn-end2end \
    --preprocessing_num_workers 8 \
    --checkpoint_dir save/llama-${model_size}-lora-json/checkpoint-6000 \
    --output_dir "save/llama-${model_size}-lora-json/eval_results" \
    --per_device_eval_batch_size 1 \
    --max_samples 100 \
    --overwrite_cache 
