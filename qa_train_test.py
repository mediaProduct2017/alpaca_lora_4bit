import json
import pprint

# Open the JSON file using the 'with' statement
with open('/data/proj_ja/ai/data/alpaca_data_cleaned_archive.json') as f:
    train_list = json.load(f)

for example in train_list[:3]:
    pprint.pprint(example)

print(len(train_list))
