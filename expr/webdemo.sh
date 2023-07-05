python ../src/web_demo_dcrn.py \
    --model_name_or_path huggyllama/llama-7b \
    --checkpoint_dir ../save/llama-7b-qlora-let/checkpoint-6000 \
    --prompt_template dcrn-end2end \

# Run with nohup:
# nohup bash webdemo.sh > webdemo.log 2>&1 &