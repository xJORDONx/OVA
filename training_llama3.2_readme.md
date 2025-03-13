# **README: Fine-Tuning LLaMA 3.2B with LoRA on RTX 4080**
---
## **📌 Project Overview**
This project fine-tunes **Meta-LLaMA 3.2B** using **LoRA (Low-Rank Adaptation)** with **8-bit quantization** for efficient training on consumer GPUs like the **RTX 4080** (12GB VRAM). The dataset used is **"RoastMe1000"**, and the training setup dynamically adjusts batch size based on available memory.

---

## **🚀 Setup & Installation**
### **🔹 Step 1: Clone Repository**
```bash
git clone <your-repo-link>
cd llama-finetuning
```

### **🔹 Step 2: Create Virtual Environment**
```bash
conda create --name exllama python=3.10 -y
conda activate exllama
```

### **🔹 Step 3: Install Dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate bitsandbytes peft
pip install sentencepiece huggingface_hub
```

### **🔹 Step 4: Enable CUDA (for NVIDIA GPU)**
Ensure you have **CUDA 12.1+** installed and check if it's working:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If it prints **True**, CUDA is enabled.

---

## **🛠️ Model Training**
### **🔹 Step 1: Set Up Environment**
Run the Python script:
```bash
python Data_Precossing.py
```
This will:
- Load the **LLaMA 3.2B** tokenizer and model.
- Load and preprocess the dataset.
- Apply **LoRA** for efficient fine-tuning.
- Dynamically set batch size based on available VRAM.

### **🔹 Step 2: Adjust Batch Size (Optional)**
If you have **more VRAM**, modify the batch size in the script:
```python
train_batch_size = 6  # Try 6 if VRAM allows
eval_batch_size = 6
grad_accum_steps = 2
```
Use **nvidia-smi** to monitor VRAM usage:
```bash
watch -n 1 nvidia-smi
```

### **🔹 Step 3: Start Training**
After running the script, confirm training when prompted:
```text
🚨 **ALL CHECKS PASSED! Ready to start fine-tuning.** 🚀
⚠️ Do you want to proceed with training? (yes/no): yes
```
---

## **📦 Model Saving & Output**
Once training completes, the model and tokenizer are saved:
- **Main Model Directory:**
  ```bash
  ./llama_finetuned/
  ```
- **Backup Directory (Optional):**
  ```bash
  C:/Users/vishu/Desktop/OVA/t_LLama_model/
  ```

---

## **⚡ Troubleshooting**
| **Issue** | **Solution** |
|-----------|-------------|
| `RuntimeError: CUDA out of memory` | Reduce batch size to **4 or 2** |
| `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'use_cache'` | Remove `use_cache` from `TrainingArguments` |
| `MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16` | Ignore, it's just a quantization warning |

---

## **📊 Expected Training Performance**
| **Batch Size** | **Approx. Training Time** |
|--------------|------------------|
| **4**  | ~5 hours |
| **6**  | ~4 hours |
| **8**  | Might crash (12GB VRAM limit) |

**Tip:** Run `nvidia-smi` before increasing batch size.

---

## **✅ Final Notes**
- **RTX 4080 (12GB) can handle batch size 6** but monitor VRAM.
- **LoRA reduces memory footprint** while keeping fine-tuning effective.
- **For faster inference**, consider converting to **GGUF** using `llama.cpp`.

Happy fine-tuning! 🚀🔥
