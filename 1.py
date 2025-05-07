# %%
# 安装和导入必要的库
# !pip install  unsloth
# %%
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

#%%
# 1. 定义KL散度计算函数 (这里是偏向反KL散度)
def compute_skewed_rkl(logits_student, logits_teacher, target_labels, padding_id,
                       reduction="sum", temp=1.0, skew_lambda=0.1):
    """计算偏向反KL散度: KL(student || mixed_distribution)
       mixed_distribution = (1-skew_lambda) * teacher + skew_lambda * student
    """
    # 温度缩放
    logits_student_scaled = logits_student / temp
    logits_teacher_scaled = logits_teacher / temp

    # 学生模型的概率和对数概率 (来自缩放后的logits)
    probs_student = torch.softmax(logits_student_scaled, dim=-1, dtype=torch.float32)
    log_probs_student = torch.log_softmax(logits_student_scaled, dim=-1, dtype=torch.float32)

    # 教师模型的概率 (来自缩放后的logits, 不应反向传播梯度)
    with torch.no_grad():
        probs_teacher = torch.softmax(logits_teacher_scaled, dim=-1, dtype=torch.float32)

    # 计算混合概率分布
    # mixed_probs = (1 - skew_lambda) * p_teacher + skew_lambda * p_student
    mixed_probs = (1 - skew_lambda) * probs_teacher + skew_lambda * probs_student
    # 防止 mixed_probs 为0导致log(0)数值问题，添加一个极小值
    mixed_log_probs = torch.log(mixed_probs + 1e-10)

    # KL散度计算: p_student * (log p_student - log p_mixed)
    kl_divergence = probs_student * (log_probs_student - mixed_log_probs)
    kl_divergence = kl_divergence.sum(dim=-1) # 在词汇表维度上求和

    # 处理padding
    if target_labels is not None and padding_id is not None:
        pad_mask = (target_labels == padding_id)
        kl_divergence.masked_fill_(pad_mask, 0.0)

    if reduction == "sum":
        kl_loss = kl_divergence.sum()
    elif reduction == "mean":
        if target_labels is not None and padding_id is not None:
            num_tokens = (target_labels != padding_id).sum()
            kl_loss = kl_divergence.sum() / num_tokens if num_tokens > 0 else torch.tensor(0.0).to(kl_divergence.device)
        else:
            kl_loss = kl_divergence.mean()
    else:
        kl_loss = kl_divergence

    return kl_loss

#%%
# 2. 定义KDTrainer (知识蒸馏训练器)
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, use_ce_loss=True,
                 kl_loss_weight=0.5, skew_lambda_rkl=0.1, # MODIFIED: Added skew_lambda_rkl
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.use_ce_loss = use_ce_loss
        self.kl_loss_weight = kl_loss_weight
        self.skew_lambda_rkl = skew_lambda_rkl # MODIFIED: Store skew_lambda for skewed_rkl
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        loss_ce_student = outputs_student.loss
        logits_student = outputs_student.logits

        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        if logits_student.shape[-1] != logits_teacher.shape[-1]:
            vocab_size_student = logits_student.shape[-1]
            logits_teacher = logits_teacher[..., :vocab_size_student]

        labels = inputs.get("labels")

        # 计算偏向反KL散度损失
        kl_loss = compute_skewed_rkl( # MODIFIED: Changed to compute_skewed_rkl
            logits_student,
            logits_teacher,
            target_labels=labels,
            padding_id=self.label_pad_token_id if hasattr(self, 'label_pad_token_id') else -100,
            temp=2.0,
            reduction="sum",
            skew_lambda=self.skew_lambda_rkl # MODIFIED: Pass skew_lambda
        )

        if self.use_ce_loss:
            total_loss = self.kl_loss_weight * kl_loss + (1 - self.kl_loss_weight) * loss_ce_student
        else:
            total_loss = kl_loss

        return (total_loss, outputs_student) if return_outputs else total_loss

#%%
# 3. 配置参数
# 模型和路径
teacher_model_path = "qwen_teacher_finetune"
student_model_name = "unsloth/Qwen2.5-0.5B"
output_dir_distillation = "./results_qwen_student_distilled_skewed_rkl" # MODIFIED
save_directory_student = "qwen_student_distilled_skewed_rkl_final" # MODIFIED

# 数据集和格式化
dataset_name = "yahma/alpaca-cleaned"
alpaca_prompt_template = """Below is an instruction that describes a task, paired with
an input that provides further context. Write a response that appropriately
completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 训练超参数
max_seq_length = 1024
load_in_4bit = True

if torch.backends.mps.is_available():
    load_in_4bit = False
    dtype = torch.float16
    print("MPS detected. Disabling 4-bit quantization and using float16.")
else:
    dtype = None
    print("CUDA or CPU detected. Using auto dtype and 4-bit quantization if enabled.")


#%%
# 蒸馏特定参数
distill_use_ce_loss = True
distill_kl_loss_weight = 0.5
distill_skew_lambda_rkl_value = 0.1 # MODIFIED: Added skew_lambda configuration for RKL
distill_epochs = 3
distill_batch_size = 2
distill_grad_accum = 8
distill_lr = 5e-5
#%%
# 4. 加载数据集和预处理
print("Loading and formatting dataset...")
dataset_full = load_dataset(dataset_name, split="train")
dataset = dataset_full

tokenizer_for_formatting = FastLanguageModel.get_tokenizer(student_model_name)
EOS_TOKEN = tokenizer_for_formatting.eos_token
if EOS_TOKEN is None:
    tokenizer_for_formatting.eos_token = "<|endoftext|>"
    EOS_TOKEN = tokenizer_for_formatting.eos_token

def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt_template.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)
print(f"Dataset formatted. Number of examples: {len(dataset)}")

#%%
# 5. 加载教师模型 (已微调)
print(f"Loading fine-tuned teacher model from {teacher_model_path}...")
teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
    model_name=teacher_model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(teacher_model)
print("Teacher model loaded.")

#%%
# 6. 加载学生模型并配置LoRA
print(f"Loading student model ({student_model_name}) and configuring LoRA...")
student_model, student_tokenizer = FastLanguageModel.from_pretrained(
    model_name=student_model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
student_model = FastLanguageModel.get_peft_model(
    student_model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("Student model loaded and LoRA configured.")
student_model.print_trainable_parameters()

if student_tokenizer.eos_token is None:
    student_tokenizer.eos_token = EOS_TOKEN
if teacher_tokenizer.eos_token is None:
    teacher_tokenizer.eos_token = EOS_TOKEN


#%%
# 7. 配置蒸馏训练参数
print("Configuring TrainingArguments for distillation...")
distill_training_args = TrainingArguments(
    output_dir=output_dir_distillation,
    num_train_epochs=distill_epochs,
    per_device_train_batch_size=distill_batch_size,
    gradient_accumulation_steps=distill_grad_accum,
    learning_rate=distill_lr,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=not is_bfloat16_supported() and not torch.backends.mps.is_available(),
    bf16=is_bfloat16_supported() and not torch.backends.mps.is_available(),
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
)

#%%
# 8. 初始化KDTrainer并开始训练
print("Initializing KDTrainer...")
distill_trainer = KDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=distill_training_args,
    train_dataset=dataset,
    tokenizer=student_tokenizer,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    use_ce_loss=distill_use_ce_loss,
    kl_loss_weight=distill_kl_loss_weight,
    skew_lambda_rkl=distill_skew_lambda_rkl_value # MODIFIED: Pass configured skew_lambda for RKL
)

print("Starting distillation training with Skewed Reverse KL Divergence...") # MODIFIED
distill_trainer.train()
print("Distillation training completed.")

#%%
# 9. 保存蒸馏后的学生模型
print(f"Saving distilled student model to {save_directory_student}...")
student_model.save_pretrained(save_directory_student)
student_tokenizer.save_pretrained(save_directory_student)
print("Distilled student model saved.")

print("\nKnowledge distillation process (Skewed Reverse KL) finished.") # MODIFIED