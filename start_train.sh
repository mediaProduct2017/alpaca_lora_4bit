export LLAMA_MODEL=/data/proj_ja/ai/llama-7b-4bit/llama-7b-4bit-128g.safetensors
export LLAMA_DIR=/data/proj_ja/ai/llama-7b-4bit/
export DATASET=/data/proj_ja/ai/data/alpaca_data_cleaned_archive.json

python finetune2.py "$DATASET" --epochs=8 --save_steps=50 --save_total_limit=2 --logging_steps=10 --val_set_size=0.1 --verbose --grad_chckpt --mbatch_size=4 --batch_size=8 --cutoff_len=1024 --groupsize=128 --xformers --llama_q4_model="$LLAMA_MODEL" --llama_q4_config_dir="$LLAMA_DIR"

# --epochs=2
# --save_steps=20
# --grad_chckpt
# --mbatch_size=8
# --cutoff_len=512
# --val_set_size=0
# --logging_steps=10
