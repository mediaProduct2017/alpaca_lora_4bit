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
chmod +x start_retrain.sh
chmod +x start_infer.sh
chmod +x start_all.sh

conda install -c conda-forge aria2
# This command will install `aria2c` from the `conda-forge` channel, which provides packages not available in the default Conda channels.

aria2c --version
# If the installation was successful, it should display the version information for `aria2c`.

./start_all.sh

./start_train.sh

./start_infer.sh

./start_retrain.sh

cd /data/proj_ja/ai/data

cp alpaca_data_cleaned_archive.json test/qa_test.json
cp alpaca_data_cleaned_archive.json train/qa_train.json

cd /data/proj_ja/ai

unzip llama-7b-4bit.zip

git checkout -- start_train.sh
git checkout -- start_retrain.sh
git checkout -- start_infer.sh
