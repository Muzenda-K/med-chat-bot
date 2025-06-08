#!/usr/bin/env python
# coding: utf-8

# # Fine-Tuning BioGPT for Medical Telegram Bot

# In[1]:


# !pip install transformers datasets peft accelerate bitsandbytes -q

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig


# -----------------------------
# Step 1: Load and Prepare Dataset
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")

# Split into train/validation (10% for validation)
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
formatted_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],
})

# Format the data as "Question: ... Answer: ..."
def format_dataset(examples):
    return {
        "text": [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(examples["Question"], examples["Answer"])
        ]
    }

formatted_dataset = formatted_dataset.map(format_dataset, batched=True,
                                          remove_columns=["Question", "Answer", "qtype"])


# -----------------------------
# Step 2: Load Tokenizer and Model
# -----------------------------
print("Loading BioGPT model and tokenizer...")
model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with trust remote code if needed
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


# -----------------------------
# Step 3: Apply LoRA Configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,                     # Low-rank matrix dimension
    lora_alpha=16,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention matrices to adapt
    lora_dropout=0.1,        # Dropout for LoRA layers
    bias="none",             # Don't train bias
    task_type="CAUSAL_LM"    # For language modeling
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show only a small percentage of parameters are trainable


# -----------------------------
# Step 4: Tokenize Dataset
# -----------------------------
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_special_tokens_mask=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True,
                                          remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# -----------------------------
# Step 5: Define Training Arguments
# -----------------------------
output_dir = "./biogpt-medquad-lora-finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch",
    group_by_length=True,
)


# -----------------------------
# Step 6: Initialize and Run Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()


# -----------------------------
# Step 7: Save the Fine-Tuned Model
# -----------------------------
# -----------------------------
# Step 7: Save the Fine-Tuned Model (Fixed)
# -----------------------------
print("Saving model with shared tensors support...")

from safetensors.torch import save_model
import os

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save only the PEFT-adapted part of the model
save_model(model, os.path.join(output_dir, "adapter_model.safetensors"))

# Save tokenizer and config for reloading
tokenizer.save_pretrained(output_dir)
model.config.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

# -----------------------------
# Step 8: Evaluate Model
# -----------------------------
eval_results = trainer.evaluate()
print(f"Evaluation loss: {eval_results['eval_loss']}")


# -----------------------------
# Step 9: Reload Model for Inference
# -----------------------------
print("Reloading model for testing...")
model = PeftModel.from_pretrained(model.base_model.model, output_dir)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()


# -----------------------------
# Step 10: Generate Responses
# -----------------------------
def generate_response(question, max_new_tokens=150):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        early_stopping=True,
        num_beams=5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("Answer:")[-1].strip()
    return answer


# -----------------------------
# Step 11: Test the Model
# -----------------------------
questions = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "What is the difference between a virus and a bacteria?",
    "Can you explain the causes of asthma?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {generate_response(q)}")
    print("-" * 50)


# In[3]:


# -----------------------------
# Step 7: Save the Fine-Tuned Model (With Config)
# -----------------------------
print("Saving model and PEFT config...")

from safetensors.torch import save_model
import os

os.makedirs(output_dir, exist_ok=True)

# Save adapter weights
save_model(model, os.path.join(output_dir, "adapter_model.safetensors"))

# Save adapter config using built-in method
model.save_pretrained(output_dir)  # This saves adapter_config.json

# Save tokenizer and base model config
tokenizer.save_pretrained(output_dir)
model.base_model.config.save_pretrained(output_dir)

print(f"Model and config saved to {output_dir}")


# In[4]:


# -----------------------------
# Step 9: Reload Model for Inference
# -----------------------------
print("Reloading model for testing...")
from peft import PeftModel, LoraConfig

# Load the base BioGPT model
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Reload the adapter
model = PeftModel.from_pretrained(base_model, output_dir)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()


# In[27]:


def generate_response(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=False,               # ← Greedy decoding
        num_beams=1,                  # ← No beam search
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("Answer:")[-1].strip()
    return clean_answer(answer)

def clean_answer(answer):
    if not answer:
        return "No answer generated."

    # Trim incomplete sentences
    ends = [".", "?", "!"]
    last_end = -1
    for e in ends:
        idx = answer.rfind(e)
        if idx > last_end:
            last_end = idx

    if last_end != -1:
        answer = answer[:last_end+1]
    else:
        answer += "..."  # Fallback

    return answer.strip()

# -----------------------------
# Step 11: Test the Model
# -----------------------------
questions = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "What is the difference between a virus and a bacteria?",
    "Can you explain the causes of asthma?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {generate_response(q)}")
    print("-" * 50)


# In[ ]:




