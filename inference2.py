# %cd /content/alpaca_lora_4bit/

import time
import torch

# from autograd_4bit import load_llama_model_4bit_low_ram_and_offload, Autograd4bitQuantLinear
# from monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_int4_lora_model
from alpaca_lora_4bit.autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_int4_lora_model
from peft import PeftModel

replace_peft_model_with_int4_lora_model()

config_path = '/data/proj_ja/ai/llama-7b-4bit/'  # "$LLAMA_DIR"
model_path = '/data/proj_ja/ai/llama-7b-4bit/llama-7b-4bit-128g.safetensors'  # "$LLAMA_MODEL"  # None

lora_path = '/data/proj_ja/alpaca_lora_4bit/alpaca_lora/checkpoint-100'
# lora_path = '/content/alpaca_lora_4bit/alpaca_lora/alpaca_lora/checkpoint-140'

# model, tokenizer = load_llama_model_4bit_low_ram_and_offload(config_path, model_path, lora_path=lora_path, groupsize=-1, seqlen=2048, max_memory=None, is_v1_model=False, bits=4)
model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, groupsize=128, device_map={'': 0})
# groupsize=-1

# groupsize=-1，表示权重量化压缩的时候，在函数Autograd4bitQuantLinear中，实际的groupsize使用in_features
# 在accelerate 0.22.0的版本中，groupsize=-1会报错，groupsize=128没问题
# groupsize=128推理效果更好，但是要慢一些

model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float32, is_trainable=False, device_map={'': 0})  # True

# Scales to half
print('Fitting 4bit scales and zeros to half')
for n, m in model.named_modules():
    if 'Autograd4bitQuantLinear' in str(type(m)) or 'Linear4bitLt' in str(type(m)):
        if hasattr(m, "is_v1_model") and m.is_v1_model:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()

# Set tokenizer
tokenizer.pad_token_id = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
# device = torch.device("auto" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from datasets import load_dataset, Dataset
import re

train_data_path = '/data/proj_ja/ai/data/alpaca_data_cleaned_archive.json'

def generate_prompt(data_point):
    return "{0}\n\n{1}\n{2}\n\n{3}\n{4}\n\n{5}".format(
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        "### Instruction:",
        data_point["instruction"],
        "### Input:",
        data_point["input"],
        "### Response:"  # ,
        # data_point["output"]
    )

dataset = load_dataset("json", data_files=train_data_path)

print(type(dataset))
print(dataset.keys())
print(len(dataset['train']))
print(type(dataset['train']))

indices_to_reserve = list()
for i, data_point in enumerate(dataset['train']):
    # print(data_point)
    if data_point['type'] == 'extract':
        indices_to_reserve.append(i)

dataset_test = dataset['train'][indices_to_reserve]
print(dataset_test.keys())
dataset_test = Dataset.from_dict(dataset_test)

print(type(dataset_test))

test_dataset = dataset_test.map(lambda x: {'text': generate_prompt(x)})
# test_dataset = dataset_test.shuffle().map(lambda x: {'text': generate_prompt(x)})

print(len(test_dataset))
print(test_dataset[0])
print(test_dataset[0]['text'])

tokenizer.padding_side='left'

pattern = r'.+{(.+)}.+{(.+)}.+{(.+)}.*'


def extract_entities(text):
    match = re.search(pattern, text)
    primary = 'no information'
    size = 'no information'
    transfer = 'no information'
    if match:
        primary = match.group(1)
        size = match.group(2)
        transfer = match.group(3)
    return {'primary': primary, 'size': size, 'transfer': transfer}

from tqdm import tqdm

# 以下是一个样本一个样本的跑，不做batch

start = time.time()
answers = list()
for i, batch in tqdm(enumerate(test_dataset), total=len(test_dataset)):
    tokenized_batch = tokenizer(batch['text'],
                                truncation=True,  # default: False
                                max_length=512,  # 1024, 512, 2048
                                return_tensors="pt")

    # Access the tokenized inputs
    input_ids = tokenized_batch["input_ids"]
    attention_mask = tokenized_batch["attention_mask"]

    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=128,  # 128, 256
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=False,  # True has negative impact
            )

        out_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # out_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        answers_batch = [extract_entities(out_text)]
        answers.extend(answers_batch)

        if i % 10 == 0:
            print(f"\n{answers_batch}")

print(answers[:5])
end = time.time()
print('Time:')
print(end - start)

correct_answers = test_dataset['output']
print(type(correct_answers))
print(len(correct_answers))
print(correct_answers[:10])

true_labels = [extract_entities(answer) for answer in correct_answers]
print(true_labels[:5])

for example in answers:
    for k,v in example.items():
        if v == 'none' or v == 'None':
            example[k] = 'no information'

print(answers[:5])

from collections import defaultdict

print(len(answers))

predicted_labels = answers

accurate = defaultdict(int)
for a, b in zip(true_labels, predicted_labels):
    for ele in ['primary', 'size', 'transfer']:
        if a[ele] == b[ele]:
            accurate[ele] += 1

for ele in ['primary', 'size', 'transfer']:
    print(f"accuracy of {ele}: {accurate[ele]/len(true_labels)}\n")

# 预测答案不为unknown时的accuracy, let's call it precision_1

accurate = defaultdict(int)
total = defaultdict(int)
for a, b in zip(true_labels, predicted_labels):
    for ele in ['primary', 'size', 'transfer']:
        if b[ele] != 'no information':
            total[ele] += 1
            if a[ele] == b[ele]:
                accurate[ele] += 1

for ele in ['primary', 'size', 'transfer']:
    print(f"precision_1 of {ele}: {accurate[ele]/total[ele]}\n")

p1 = accurate['primary']/total['primary']

# 预测答案为unknown时的accuracy, let's call it precision_2

accurate = defaultdict(int)
total = defaultdict(int)
for a, b in zip(true_labels, predicted_labels):
    for ele in ['primary', 'size', 'transfer']:
        if b[ele] == 'no information':
            total[ele] += 1
            if a[ele] == b[ele]:
                accurate[ele] += 1

for ele in ['primary', 'size', 'transfer']:
    print(f"precision_2 of {ele}: {accurate[ele]/total[ele]}\n")

# 实际答案不为unknown时的accuracy, let's call it recall_1
# 这个实际上是抽取任务

accurate = defaultdict(int)
total = defaultdict(int)
for a, b in zip(true_labels, predicted_labels):
    for ele in ['primary', 'size', 'transfer']:
        if a[ele] != 'no information':
            total[ele] += 1
            if a[ele] == b[ele]:
                accurate[ele] += 1

for ele in ['primary', 'size', 'transfer']:
    print(f"recall_1 of {ele}: {accurate[ele]/total[ele]}\n")

r1 = accurate['primary']/total['primary']

# 实际答案为unknown时的accuracy, let's call it recall_2
# 这个带有一定的推理任务

accurate = defaultdict(int)
total = defaultdict(int)
for a, b in zip(true_labels, predicted_labels):
    for ele in ['primary', 'size', 'transfer']:
        if a[ele] == 'no information':
            total[ele] += 1
            if a[ele] == b[ele]:
                accurate[ele] += 1

for ele in ['primary', 'size', 'transfer']:
    print(f"recall_2 of {ele}: {accurate[ele]/total[ele]}\n")

# 原发部位的f1_score1
f1 = 2*(p1*r1)/(p1+r1)
print(f1)
