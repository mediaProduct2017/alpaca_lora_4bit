import os
import sys
# set src so alpaca_lora_4bit package is available without installing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# project_root = os.path.abspath(os.getcwd())
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

# Early load config to replace attn if needed
from alpaca_lora_4bit.arg_parser import get_config
ft_config = get_config()

from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_int4_lora_model
replace_peft_model_with_int4_lora_model()

if ft_config.flash_attention:
    from alpaca_lora_4bit.monkeypatch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
elif ft_config.xformers:
    from alpaca_lora_4bit.monkeypatch.llama_attn_hijack_xformers import hijack_llama_attention
    hijack_llama_attention()

from alpaca_lora_4bit import autograd_4bit
if ft_config.backend.lower() == 'triton':
    autograd_4bit.switch_backend_to('triton')
else:
    autograd_4bit.switch_backend_to('cuda')

import sys
import os

import peft
import peft.tuners.lora

import wandb
import torch
import transformers
from alpaca_lora_4bit.autograd_4bit import load_llama_model_4bit_low_ram
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel, set_peft_model_state_dict

from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging
from transformers import GenerationConfig

logging.set_verbosity_warning()
logger = logging.get_logger("transformers")

# Define a custom callback class
class CustomCallback(TrainerCallback):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.config = GenerationConfig.from_model_config(model.config)
        self.config.max_length = 2048  # 4096
        self.config.max_new_tokens = 160  # 128, 256
        self.config.do_sample=False
        self.config.return_dict_in_generate=True
        self.config.output_scores=False

        print('The default generation config of the model:')
        print(model.generation_config)

        print('Finish init of CustomCallback')

    def on_init_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform any custom initialization tasks before training starts
        return control

    def on_train_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform any custom tasks at the beginning of training
        return control

    def on_epoch_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform any custom tasks at the beginning of training
        return control

    def on_step_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform any custom tasks at the beginning of training
        return control

    def infer_example(self, example):

        # inputs = self.tokenizer(example, return_tensors="pt").to(self.model.device)
        # 暂时没用

        outputs = self.model.generate(
            input_ids=self.tokenizer.encode(example, return_tensors="pt").to(self.model.device),
            # **inputs,
            # generation_config=self.config,
            do_sample=False,
            max_new_tokens=128,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=False,  # True has negative impact
            )
        out_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # out_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print the inference result
        print(f"Inference result:\n{out_text}")

    def generate_prompt(self, question):
        return "{0}\n\n{1}\n{2}\n\n{3}\n{4}\n\n{5}".format(
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
            "### Instruction:",
            question,
            "### Input:",
            "",
            "### Response:"
        )

    def generate_prompt2(self, question, input):
        return "{0}\n\n{1}\n{2}\n\n{3}\n{4}\n\n{5}".format(
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
            "### Instruction:",
            question,
            "### Input:",
            input,
            "### Response:"
        )

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform any custom tasks at the end of each training step; 20, 5
        if trainer.state.global_step > 1 and (trainer.state.global_step-1) % 200 == 0:
        # if (trainer.state.global_step-1) % 20 == 0:  # 20
            print(f"\n\nCustom content at step {trainer.state.global_step-1}\n")
            logger.info("INFO\n")
            logger.warning("WARN\n")

            # Print the inference results
            self.infer_example(self.generate_prompt("什么是肿瘤转移"))
            # self.infer_example(self.generate_prompt("肿瘤转移是什么"))
            self.infer_example(self.generate_prompt("what is tumor metastasis"))
            # self.infer_example("what is tumor metastasis")
            # self.infer_example(self.generate_prompt("脑细胞能移动吗"))
            self.infer_example(self.generate_prompt("脑细胞会移动？"))
            # self.infer_example("脑细胞会移动？")
            # self.infer_example("Can brain cells move? By movement I mean long distance migration (preferably within the brain only).")
            # self.infer_example(self.generate_prompt("Can brain cells move? By movement I mean long distance migration (preferably within the brain only)."))

            # self.infer_example(self.generate_prompt("You are given a question on professional law. You are also given 4 answer options (associated with \"A\", \"B\", \"C\", \"D\"), out of which only one is correct. You need to answer the question by selecting the correct option. You should only answer with the choice letter, not the whole answer. \n\nOne afternoon, a pilot was flying a small airplane when it suddenly ran out of gas. As he was coming in for an emergency landing, the plane crossed into a neighboring state at a very low altitude. At this time, a 9-year-old boy was walking to school when he was struck and injured by an object, which may have fallen from the plane. In federal court, a negligence suit was brought against the pilot by the father of the boy for his son. Accompanied by his father, the boy had visited an attorney for preliminary discussions regarding the case. However, the father did not retain the attorney to represent his son in the lawsuit. Instead, the father hired another lawyer to handle the case. At trial, the pilot's attorney calls the consulting attorney to testify what the boy had said to him regarding his physical condition during the consultation that the attorney had had with the boy and his father. The attorney's testimony is\n\n(A)admissible, because the attorney-client privilege was waived by the filing of the lawsuit. \n(B)admissible, because there is no privilege of confidentiality when a person other than the client is present at the attorney-client consultation. \n(C)inadmissible, because the attorney-client privilege prevents such a breach of confidential communications. \n(D)inadmissible, because it was a statement of physical condition not made for the purpose of obtaining medical treatment."))
            # self.infer_example(self.generate_prompt("You are given a question on professional law. You are also given 2 answer options (associated with \"A\", \"B\"), out of which only one is correct. You need to answer the question by selecting the correct option, and give explanations. \n\nOne afternoon, a pilot was flying a small airplane when it suddenly ran out of gas. As he was coming in for an emergency landing, the plane crossed into a neighboring state at a very low altitude. At this time, a 9-year-old boy was walking to school when he was struck and injured by an object, which may have fallen from the plane. In federal court, a negligence suit was brought against the pilot by the father of the boy for his son. Accompanied by his father, the boy had visited an attorney for preliminary discussions regarding the case. However, the father did not retain the attorney to represent his son in the lawsuit. Instead, the father hired another lawyer to handle the case. At trial, the pilot's attorney calls the consulting attorney to testify what the boy had said to him regarding his physical condition during the consultation that the attorney had had with the boy and his father. The attorney's testimony is\n\n(A)admissible. \n(B)inadmissible.\n"))
            # self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis. Choose the correct option from the following options, and give explanations.\nOptions: (A) Tumor metastasis present. (B) No tumor metastasis present.\nThe report is:", "Combining clinical observations, after rectal cancer surgery, there have been no significant changes in the bilateral lung metastases compared to before June 9, 2016. There is slight thickening of the bilateral pleura, with no significant changes. There is a low-density lesion in the thyroid, with no significant changes. Multiple nodules and masses are present near the pleura of both lungs, some showing lobulated and rough edges, causing traction on adjacent pleura. The largest one is located in the lower lobe of the left lung, with a longitudinal diameter of approximately 3.0 cm (mediastinal window). It shows small air-filled cavities internally and exhibits heterogeneous enhancement on contrast-enhanced imaging. No enlarged lymph nodes are observed in the bilateral hila or mediastinum. There is slight thickening of the bilateral pleura. No signs of pleural effusion are seen in both pleural cavities. A low-density lesion is noted in the right lobe of the thyroid, with relatively clear margins."))
            # self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis. Choose the correct option from the following options, and give explanations.\nOptions: (A) Tumor metastasis present. (B) No tumor metastasis present.\nThe report is:", "A round mass shadow with a diameter of approximately 0.56 cm is seen in the dorsal segment of the lower lobe of the right lung. Scattered patchy shadows are present in the lower right lung. There are no signs of pleural effusion in both lung cavities. No abnormalities are observed in the heart and major blood vessels. There are scattered inflammations in the lower right lung."))
            # self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis. Choose the correct option from the following options, and give explanations.\nOptions: (A) Tumor metastasis present. (B) No tumor metastasis present.\nThe report is:", "An irregular soft tissue density lesion is visible in the upper lobe of the left lung, measuring approximately 2.6 cm × 3.7 cm. On contrast-enhanced scan, it shows significant heterogeneous enhancement with shallow lobulated margins. There are scattered multiple nodular lesions within both lung fields, with the largest one measuring approximately 0.6 cm in diameter. Multiple enlarged lymph nodes are observed in the 4R and 7 regions of the mediastinum, with the largest one measuring approximately 1.0 cm in short axis. There is a localized increase in bone density observed in the right side of the T6 vertebral body, right pedicle, and the posterior aspect of the 6th rib.\n1. Left upper lobe cancer involving the pleura and multiple metastases in both lungs. 2. Enlarged lymph nodes in the mediastinum. 3. Localized increase in bone density in the right side of the T6 vertebral body, right pedicle, and the posterior aspect of the 6th rib. Please consider other examinations for further evaluation."))
            # self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis. Choose the correct option from the following options, and give explanations.\nOptions: (A) Tumor metastasis present. (B) No tumor metastasis present.\nThe report is:", "Heart structure and movement are normal. No abnormal blood flow signals were detected in the chambers, ventricles, or major blood vessels. Normal heart function was observed on echocardiogram with the following values: - Left Atrium (LA): 28mm - Aorta (AO): 27mm - Mitral Valve E Velocity (MV E): 0.55m/s - Ejection Fraction (EF): 56% - Left Ventricle (LV): 45mm - Interventricular Septum (IVS): 11mm - Aortic Velocity (A): 0.57m/s - Fractional Shortening (FS): 29% - Right Atrium (RA): 28mm - Left Ventricular Posterior Wall (LVPW): 8mm - Right Ventricle (RV): 17mm - Pulmonary Artery (PA): 18mm \nTwo-dimensional and M-mode echocardiographic features: - Normal ascending aorta diameter with a smooth wall, normal amplitude of the main wave, and presence of a dicrotic wave. - Normal pulmonary artery diameter. - Normal dimensions of the chambers. - Normal ventricular wall thickness and movement. - Intact continuity of the atrial and ventricular septum. - Normal morphology, structure, and movement of the valves. - No abnormalities detected in the pericardium or pericardial cavity. - No abnormalities found in color and spectral Doppler echocardiography."))

            self.infer_example(self.generate_prompt2("According to the following Chinese tumor imaging report, extract the primary tumor site, tumor size and metastasis site.\nThe report is:", "1.结合临床,直肠癌术后,双肺转移瘤,较前2016-6-9变化不著 2.双侧胸膜略增厚,变化不著 4.甲状腺低密度灶,变化不著。双肺近胸膜下示多个结节灶及肿块,部分边缘分叶、毛糙,牵拉邻近胸膜,大者位于左肺下叶,长径约3.0CM(纵隔窗),内示小空泡影,增强呈不均质强化。双侧肺门及纵隔未见增大淋巴结。双侧胸膜略增厚。双侧胸腔未见积液征象。甲状腺右叶见低密度灶,边缘较清晰。\t"))
            self.infer_example(self.generate_prompt2("According to the following Chinese tumor imaging report, extract the primary tumor site, tumor size and metastasis site.\nThe report is:", "左肺上叶可见一不规则软组织密度灶,约2.6CM×3.7CM,增强扫描呈明显不均匀强化,其边缘呈浅分叶,双肺野内散在多发结节灶,大者直径约0.6CM,纵隔内4R、7区示多发肿大淋巴结,大者短径约1.0CM。T6椎体右侧椎弓根及右侧第6后肋可见局限性骨质密度增高影。\t1.左肺上叶癌累及胸膜并双肺多发转移2.纵隔淋巴结肿大3.T6椎体右侧椎弓根及右侧第6后肋局限性骨质密度增高,请结合其他检查。"))

            # self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis. Choose only one correct option from the following options and answer with the corresponding choice letter.\nOptions: A. No metastasis present; B. Metastasis present.\nThe tumor imaging report is:", "1.结合临床,直肠癌术后,双肺转移瘤,较前2016-6-9变化不著 2.双侧胸膜略增厚,变化不著 4.甲状腺低密度灶,变化不著。双肺近胸膜下示多个结节灶及肿块,部分边缘分叶、毛糙,牵拉邻近胸膜,大者位于左肺下叶,长径约3.0CM(纵隔窗),内示小空泡影,增强呈不均质强化。双侧肺门及纵隔未见增大淋巴结。双侧胸膜略增厚。双侧胸腔未见积液征象。甲状腺右叶见低密度灶,边缘较清晰。\t"))
            # self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis.\nThe tumor imaging report is:", "1.结合临床,直肠癌术后,双肺转移瘤,较前2016-6-9变化不著 2.双侧胸膜略增厚,变化不著 4.甲状腺低密度灶,变化不著。双肺近胸膜下示多个结节灶及肿块,部分边缘分叶、毛糙,牵拉邻近胸膜,大者位于左肺下叶,长径约3.0CM(纵隔窗),内示小空泡影,增强呈不均质强化。双侧肺门及纵隔未见增大淋巴结。双侧胸膜略增厚。双侧胸腔未见积液征象。甲状腺右叶见低密度灶,边缘较清晰。\t"))
            # self.infer_example(self.generate_prompt2("Based on the tumor imaging report provided, please select the appropriate choice from the options below to indicate the presence or absence of tumor metastasis. Please respond with the corresponding letter:\nOptions: A. Metastasis present; B. No metastasis present.\nThe report is:", "1.结合临床,直肠癌术后,双肺转移瘤,较前2016-6-9变化不著 2.双侧胸膜略增厚,变化不著 4.甲状腺低密度灶,变化不著。双肺近胸膜下示多个结节灶及肿块,部分边缘分叶、毛糙,牵拉邻近胸膜,大者位于左肺下叶,长径约3.0CM(纵隔窗),内示小空泡影,增强呈不均质强化。双侧肺门及纵隔未见增大淋巴结。双侧胸膜略增厚。双侧胸腔未见积液征象。甲状腺右叶见低密度灶,边缘较清晰。\t"))

            # self.infer_example("肿瘤转移是指原发肿瘤细胞离开原处，通过血液或淋巴传播到其他身体部位。\n根据下面的肿瘤医学影像报告，请回答，是否存在肿瘤转移？从下列选项中选择唯一一个正确选项并只回答对应的选项。\n可选选项为：A 存在转移 B 不存在转移。\n肿瘤医学影像报告为：\n右乳术后缺如,部分胸肌存在,术区胸壁及皮肤局部略增厚;右侧腋窝术后,结构紊乱。左乳未见明确异常。右侧内乳区、左侧腋窝可见小淋巴结,大者短径不足0.5CM。右肺中叶见一结节灶,长径约1.2CM。右肺中叶可见类圆形囊状过度充气区及条片影。双肺门及纵隔内未见肿大淋巴结。右侧胸膜略增厚。扫描野内肝实质密度减低。脾内示颗粒状致密影。胆囊腔内密度略增高。  右侧部分肋骨密度增高。\t1.右乳癌术后改变,术区胸壁及皮肤局部略增厚,较前(2016-2-26)基本变化不著 2.考虑右肺转移,略增大;右肺中叶含气囊肿、纤维灶,变化不著 3.脂肪肝;脾内钙化灶 4.右侧部分肋骨密度增高,变化不著。")

        return control

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform any custom tasks at the end of training
        return control

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform any custom tasks at the end of training

        # Print the inference results
        self.infer_example(self.generate_prompt("什么是肿瘤转移"))
        self.infer_example(self.generate_prompt("肿瘤转移是什么"))
        self.infer_example(self.generate_prompt("what is tumor metastasis"))
        self.infer_example(self.generate_prompt("脑细胞能移动吗"))
        self.infer_example(self.generate_prompt("脑细胞会移动？"))
        self.infer_example("Can brain cells move? By movement I mean long distance migration (preferably within the brain only).")
        self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis. Choose only one correct option from the following options and answer with the corresponding choice letter.\nOptions: A. No metastasis present; B. Metastasis present.\nThe tumor imaging report is:", "1.结合临床,直肠癌术后,双肺转移瘤,较前2016-6-9变化不著 2.双侧胸膜略增厚,变化不著 4.甲状腺低密度灶,变化不著。双肺近胸膜下示多个结节灶及肿块,部分边缘分叶、毛糙,牵拉邻近胸膜,大者位于左肺下叶,长径约3.0CM(纵隔窗),内示小空泡影,增强呈不均质强化。双侧肺门及纵隔未见增大淋巴结。双侧胸膜略增厚。双侧胸腔未见积液征象。甲状腺右叶见低密度灶,边缘较清晰。\t"))
        self.infer_example(self.generate_prompt2("According to the following tumor imaging report, answer whether there is tumor metastasis.\nThe tumor imaging report is:", "1.结合临床,直肠癌术后,双肺转移瘤,较前2016-6-9变化不著 2.双侧胸膜略增厚,变化不著 4.甲状腺低密度灶,变化不著。双肺近胸膜下示多个结节灶及肿块,部分边缘分叶、毛糙,牵拉邻近胸膜,大者位于左肺下叶,长径约3.0CM(纵隔窗),内示小空泡影,增强呈不均质强化。双侧肺门及纵隔未见增大淋巴结。双侧胸膜略增厚。双侧胸腔未见积液征象。甲状腺右叶见低密度灶,边缘较清晰。\t"))
        self.infer_example(self.generate_prompt2("Based on the tumor imaging report provided, please select the appropriate choice from the options below to indicate the presence or absence of tumor metastasis. Please respond with the corresponding letter:\nOptions: A. Metastasis present; B. No metastasis present.\nThe report is:", "1.结合临床,直肠癌术后,双肺转移瘤,较前2016-6-9变化不著 2.双侧胸膜略增厚,变化不著 4.甲状腺低密度灶,变化不著。双肺近胸膜下示多个结节灶及肿块,部分边缘分叶、毛糙,牵拉邻近胸膜,大者位于左肺下叶,长径约3.0CM(纵隔窗),内示小空泡影,增强呈不均质强化。双侧肺门及纵隔未见增大淋巴结。双侧胸膜略增厚。双侧胸腔未见积液征象。甲状腺右叶见低密度灶,边缘较清晰。\t"))

        return control

    def on_log(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        # Perform any custom tasks during logging
        return control

    def on_save(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        # Perform any custom tasks during logging
        return control

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        # Perform any custom tasks during logging
        return control

    def on_prediction_step(self, args, state, control, model=None, inputs=None, **kwargs):
        # Perform any custom tasks during the prediction step of evaluation
        return control

# ! Config
from alpaca_lora_4bit import train_data

# * Show loaded parameters
if ft_config.local_rank == 0:
    print(f"{ft_config}\n")

if ft_config.gradient_checkpointing:
    print('Disable Dropout.')

if ft_config.mbatch_size > ft_config.batch_size:
    raise Exception('batch_size need to be larger than mbatch_size.')

# Load Basic Model
model, tokenizer = load_llama_model_4bit_low_ram(ft_config.llama_q4_config_dir,
                                                  ft_config.llama_q4_model,
                                                  device_map=ft_config.device_map,
                                                  groupsize=ft_config.groupsize,
                                                  is_v1_model=ft_config.v1)

# Config Lora
lora_config = LoraConfig(
    r=ft_config.lora_r,
    lora_alpha=ft_config.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=ft_config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
if ft_config.lora_apply_dir is None:
    model = get_peft_model(model, lora_config)
else:
    device_map = ft_config.device_map
    if ft_config.ddp:
        device_map = {'': 0}
    else:
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            device_map = {'': 0}
    print('Device map for lora:', device_map)
    model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, device_map=device_map, torch_dtype=torch.float32, is_trainable=True)
    print(ft_config.lora_apply_dir, 'loaded')


# Scales to half
print('Fitting 4bit scales and zeros to half')
for n, m in model.named_modules():
    if 'Autograd4bitQuantLinear' in str(type(m)) or 'Linear4bitLt' in str(type(m)):
        if hasattr(m, "is_v1_model") and m.is_v1_model:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()

# Set tokenizer
tokenizer.pad_token_id = 0

if not ft_config.skip:
    # Load Data
    data = None
    if ft_config.ds_type == "txt" and not ft_config.skip:
        #### LLaMa
        data = train_data.TrainTxt(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "alpaca" and not ft_config.skip:
        #### Stanford Alpaca-like Data
        data = train_data.TrainSAD(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "gpt4all" and not ft_config.skip:
        #### GPT4All Data
        data = train_data.TrainGPT4All(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "bluemoon" and not ft_config.skip:
        #### Blue Moon Data
        data = train_data.TrainBlueMoon(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    else:
        raise NotImplementedError("ERROR: Unknown dataset format")
    data.prepare_data(thd=ft_config.txt_row_thd, use_eos_token=ft_config.use_eos_token)
    ####

    # Use gradient checkpointing
    if ft_config.gradient_checkpointing:
        print('Applying gradient checkpointing ...')
        from alpaca_lora_4bit.gradient_checkpointing import apply_gradient_checkpointing
        apply_gradient_checkpointing(model, checkpoint_ratio=ft_config.gradient_checkpointing_ratio)

    # Disable Trainer's DataParallel for multigpu
    if not ft_config.ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Count eval count for wandb
    if ft_config.val_set_size > 0:
        eval_count = 10
        eval_steps = max(
            ft_config.logging_steps, (len(data.train_data) + len(data.val_data)) // (eval_count*ft_config.mbatch_size)
        )
        print(f"This is useless: Run eval every {eval_steps} steps")
    else:
        eval_steps = 0

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=ft_config.mbatch_size,
        gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
        warmup_steps=ft_config.warmup_steps,
        optim="adamw_torch",
        num_train_epochs=ft_config.epochs,
        learning_rate=ft_config.lr,
        fp16=True,
        logging_steps=ft_config.logging_steps,
        evaluation_strategy="steps" if eval_steps != 0 else "no",
        save_strategy="steps",
        eval_steps=100 if eval_steps != 0 else None,  # 100, 5
        save_steps=ft_config.save_steps,
        output_dir=ft_config.lora_out_dir,
        save_total_limit=ft_config.save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ft_config.ddp else None,
    )

    # 画蛇添足，多此一举
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data.train_data,
        eval_dataset=data.val_data,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[CustomCallback(model, tokenizer)],
    )
    model.config.use_cache = False

    # Set Model dict
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))

    # Set Verbose
    if ft_config.verbose:
        transformers.logging.set_verbosity_info()

    # Run Trainer
    with wandb.init(project="alpaca_lora_4bit") as run:
        if ft_config.resume_checkpoint:
            print('Resuming from {} ...'.format(ft_config.resume_checkpoint))
            import transformers.trainer
            transformers.trainer.WEIGHTS_NAME = 'adapter_model.bin'
            state_dict_peft = torch.load(os.path.join(ft_config.resume_checkpoint, 'adapter_model.bin'), map_location='cpu')
            set_peft_model_state_dict(model, state_dict_peft)
            trainer.train(resume_from_checkpoint=ft_config.resume_checkpoint)
        else:
            trainer.train()

    # Restore old model state dict
    # model.state_dict = old_state_dict

    print('Train completed.')

# Save Model
model.save_pretrained(ft_config.lora_out_dir)

if ft_config.checkpoint:
    print("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")

print('Model Saved.')
