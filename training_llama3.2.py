import torch
import os
import psutil
from datasets import load_dataset, DatasetDict
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig  
)
from peft import LoraConfig, get_peft_model, TaskType

# ✅ Check for the latest checkpoint
checkpoint_dir = "./llama_finetuned"
if os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(checkpoint_dir, max(checkpoints, key=lambda x: int(x.split('-')[-1])))
        print(f"🟢 Resuming training from: {last_checkpoint}")
    else:
        last_checkpoint = None
        print("🟡 No checkpoint found. Training will start from scratch.")
else:
    last_checkpoint = None
    print("🟡 No previous training found. Starting fresh.")

# ==============================
# ✅ Step 1: System & Environment Checks
# ==============================

print("\n🚀 **Checking System & Environment Setup...**\n")

# 🔹 Check PyTorch Version
print(f"🟢 PyTorch Version: {torch.__version__}")

# 🔹 Check CUDA Availability
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"🟢 CUDA Available: ✅ | Device: {torch.cuda.get_device_name(0)}")
    print(f"   ➡ CUDA Version: {torch.version.cuda}")
else:
    print("🔴 CUDA NOT AVAILABLE! Training will be on CPU (slower).")

# 🔹 Check Available GPU Memory
gpu_memory = torch.cuda.mem_get_info()[0] / 1e9 if cuda_available else 0
print(f"🟢 Free GPU Memory: {gpu_memory:.2f} GB" if cuda_available else "🔴 No GPU detected!")

# 🔹 Check Available CPU RAM
free_ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
print(f"🟢 Free RAM Available: {free_ram:.2f} GB")

print("\n✅ **System Checks Complete!**\n")

# ==============================
# ✅ Step 2: Load Tokenizer & Dataset
# ==============================

print("\n🚀 **Loading Tokenizer & Dataset...**\n")

# 🏗 Load LLaMA Tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, legacy=False)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# 📥 Load Dataset from Hugging Face
dataset = load_dataset("gus-gustavo/RoastMe1000")

# 🔹 Convert Dataset to DataFrame
df = dataset["train"].to_pandas()
df.rename(columns={'output': 'sentence'}, inplace=True)
df = df[['sentence']].dropna()  # Drop missing values

# 🔄 Convert DataFrame back to Hugging Face Dataset
hf_dataset = dataset["train"].from_pandas(df)

# 🔄 Split into Training (90%) & Validation (10%)
split_dataset = hf_dataset.train_test_split(test_size=0.1)

# ✅ Tokenization Function
def tokenize_function(examples):
    tokens = tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()  # ✅ Assign labels correctly
    return tokens

# 🚀 Tokenization Process
tokenized_train_dataset = split_dataset["train"].map(tokenize_function, batched=True)
tokenized_val_dataset = split_dataset["test"].map(tokenize_function, batched=True)

# 📦 Wrap into DatasetDict
dataset = DatasetDict({
    "train": tokenized_train_dataset,
    "validation": tokenized_val_dataset
})

print("\n✅ **Dataset Successfully Loaded & Tokenized!** 🚀")

# ==============================
# ✅ Step 3: Model Setup (LoRA + Quantization)
# ==============================

print("\n🚀 **Setting Up Model for Training...**\n")

# ✅ Configure Model with 8-bit Quantization & CPU RAM Offloading
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head", "embed_tokens"]
)

# ✅ Load Pretrained LLaMA Model with Quantization
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="./offload",
    offload_state_dict=True,
)

# ⚙️ Apply LoRA for Efficient Fine-Tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

# ✅ Apply LoRA for Efficient Fine-Tuning
model = get_peft_model(model, lora_config)

# ✅ **Ensure LoRA Weights Require Gradients**
for name, param in model.named_parameters():
    if "lora" in name or "adapter" in name:
        param.requires_grad = True

# ✅ Set Model to Training Mode
model.train()

# ✅ Print Trainable Parameters
model.print_trainable_parameters()

print("\n✅ **Model Setup Complete!** 🚀")

# ==============================
# ✅ Step 4: Training Setup (Optimized)
# ==============================

print("\n🚀 **Starting Training Setup...**\n")

# Dynamically adjust batch sizes based on available memory
if gpu_memory >= 10:
    train_batch_size = 5
    eval_batch_size = 4
    grad_accum_steps = 4
elif gpu_memory >= 6:
    train_batch_size = 2
    eval_batch_size = 2
    grad_accum_steps = 4
else:
    train_batch_size = 1
    eval_batch_size = 1
    grad_accum_steps = 16

print(f"\n🟢 Using Batch Size: {train_batch_size} | Gradient Accumulation Steps: {grad_accum_steps}")

# ✅ Training Arguments
training_args = TrainingArguments(
    gradient_checkpointing=False,
    output_dir="./llama_finetuned",
    per_device_train_batch_size=train_batch_size,  
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    num_train_epochs=2,
    save_steps=1000,  
    save_total_limit=2,  
    eval_strategy="steps",
    eval_steps=1000,
    logging_dir="./logs",
    fp16=True,
    bf16=False,
    optim="adamw_bnb_8bit",
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=200,
    lr_scheduler_type="cosine_with_restarts",
    label_names=["labels"]
)

# 🏋️ Fine-Tune LLaMA Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# ✅ Resume from Checkpoint
if last_checkpoint:
    print(f"🟢 Resuming training from checkpoint: {last_checkpoint}")
    
    # 🔥 Securely Load Weights Only
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ✅ Override PyTorch default restore location (secure fix)
    torch.serialization.default_restore_location = lambda storage, loc: storage
else:
    print("🟡 No checkpoint found. Starting training from scratch.")
    trainer.train()



# Define save paths
main_save_path = "./llama_finetuned"
backup_save_path = "C:/Users/vishu/Desktop/OVA/t_LLama_model"

# 💾 Save Model & Tokenizer
model.save_pretrained(main_save_path)
tokenizer.save_pretrained(main_save_path)
model.save_pretrained(backup_save_path)
tokenizer.save_pretrained(backup_save_path)

print(f"\n🎉 **Fine-Tuned Model Saved!** 🚀\n")
print(f"✅ Model available at: {main_save_path}")
print(f"✅ Model copy for sharing saved at: {backup_save_path}")
