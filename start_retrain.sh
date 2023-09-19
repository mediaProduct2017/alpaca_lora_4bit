export LLAMA_MODEL=/data/proj_ja/ai/llama-7b-4bit/llama-7b-4bit-128g.safetensors
export LLAMA_DIR=/data/proj_ja/ai/llama-7b-4bit/
export DATASET=/data/proj_ja/ai/data/alpaca_data_cleaned_archive.json

python finetune2.py "$DATASET" --resume_checkpoint=alpaca_lora/checkpoint-1376 --epochs=12 --save_steps=86 --save_total_limit=4 --logging_steps=10 --val_set_size=0 --verbose --grad_chckpt --mbatch_size=8 --batch_size=8 --cutoff_len=1024 --groupsize=128 --xformers --llama_q4_model="$LLAMA_MODEL" --llama_q4_config_dir="$LLAMA_DIR"

# --epochs=2, 8, 16, 32; 12
# --save_steps=20, 50 (default), 26; 86
# save_steps is determined by the configuration of resume_checkpoint
# --resume_checkpoint=alpaca_lora/checkpoint-100
