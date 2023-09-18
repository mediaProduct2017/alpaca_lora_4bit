import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# DATA = "/content/ai/data"
DATA = os.path.join(project_root, "ai/data")

os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/mediaProduct2017/blog_english/main/info_extract/qa_test.json -d {DATA} -o alpaca_data_cleaned_archive.json")
