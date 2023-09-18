import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# REPO = "/content/ai/llama-7b-4bit"
REPO = os.path.join(project_root, "ai/llama-7b-4bit")

# gptq model

repo_name = 'Neko-Institute-of-Science/LLaMA-7B-4bit-128g'
model_name = 'llama-7b-4bit-128g.safetensors'

os.system(f"apt -y install -qq aria2")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/{repo_name}/resolve/main/{model_name} -d {REPO} -o {model_name}")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/{repo_name}/resolve/main/tokenizer.model -d {REPO} -o tokenizer.model")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/{repo_name}/raw/main/config.json -d {REPO} -o config.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/{repo_name}/raw/main/generation_config.json -d {REPO} -o generation_config.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/{repo_name}/raw/main/special_tokens_map.json -d {REPO} -o special_tokens_map.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/{repo_name}/raw/main/tokenizer_config.json -d {REPO} -o tokenizer_config.json")
os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/{repo_name}/raw/main/tokenizer.json -d {REPO} -o tokenizer.json")


# DATA = "/content/ai/data"
DATA = os.path.join(project_root, "ai/data")

os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/mediaProduct2017/blog_english/main/info_extract/qa_test.json -d {DATA} -o alpaca_data_cleaned_archive.json")

# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
print(cwd)
