# Import libraries
import pandas as pd
import numpy as np
import os
import random
import torch
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook as tqdmnb
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers import get_cosine_schedule_with_warmup
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    logging,
    EarlyStoppingCallback,
    pipeline,
)
from peft import LoraConfig, PeftModel
from datasets import load_dataset as hf_load_dataset, Dataset
from trl import SFTTrainer
import warnings
from huggingface_hub import login

login(api_key = "****************")

# Define the directory where checkpoints are saved during training
output_dir = 'model/Nvidia_LLaMA_3_Qlora_customQA_Sample'

def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        raise ValueError(f"Directory {output_dir} does not exist")
    # List and filter checkpoint directories
    checkpoints = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and re.match(r'checkpoint-(\d+)', d)]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in directory {output_dir}")

    # Sort checkpoints numerically by the extracted number
    checkpoints.sort(key=lambda x: int(re.match(r'checkpoint-(\d+)', x).group(1)), reverse=True)
    return os.path.join(output_dir, checkpoints[0])

try:
    last_checkpoint = get_last_checkpoint(output_dir)
    print(f"Last checkpoint: {last_checkpoint}")
except Exception as e:
    print(f"Error: {e}")

print(f"Last checkpoint: {last_checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)

# Load the model
model = AutoModelForCausalLM.from_pretrained(last_checkpoint)
# Set the model to evaluation mode
model.eval()

# Define the system message
system_message = (
    "System: This is a chat between a user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
    "The assistant should also indicate when the answer cannot be found in the context."
)

warnings.filterwarnings("ignore", category=UserWarning)
print("INFERENCE (8 epochs): Llama-3 (nvidia/Llama3-ChatQA-1.5-8B)\n\n")

context = ""
while True:
    user_input = input("Enter your question (or type 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break

    if context == "":
        context_input = input("Enter context (or press Enter to skip): ").strip()
        context = context_input if context_input else "No context provided."

    # Tokenize input
    input_text = f"{system_message}\n\n{context}\n\nUser: {user_input}\n\nAssistant:"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)["input_ids"]

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            max_length=512,
            num_return_sequences=1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    print(f"Assistant: {response}\n")