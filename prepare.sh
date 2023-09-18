# 0911 original version, qa_test data


conda activate /data/proj_ja/conda_envs/llama_lora/py39

cd /data/proj_ja/

git clone https://github.com/mediaProduct2017/alpaca_lora_4bit.git

cd alpaca_lora_4bit

git fetch origin dev-ja

git checkout dev-ja

pip install -r requirements.txt -i https://mirrors.sjtug.sjtu.edu.cn/pypi/web/simple

# pip install .

chmod +x start_download.sh
chmod +x start_train.sh
chmod +x start_all.sh

./start_all.sh

./start_train.sh
