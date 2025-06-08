import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
model_name = "microsoft/biogpt"
output_dir = "./biogpt-medquad-lora-finetuned"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

print("Loading PEFT adapter...")
model = PeftModel.from_pretrained(base_model, output_dir)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Clean Answer Post-Processing
# -----------------------------
def clean_answer(answer):
    if not answer:
        return "No answer generated."
    ends = [".", "?", "!"]
    last_end = -1
    for e in ends:
        idx = answer.rfind(e)
        if idx > last_end:
            last_end = idx
    if last_end != -1:
        answer = answer[:last_end+1]
    else:
        answer += "..."
    return answer.strip()

# -----------------------------
# Generate Function for Gradio
# -----------------------------
def generate_response(question):
    prompt = f"Question: {question}\nAnswer briefly in one paragraph:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=120,
        do_sample=False,
        num_beams=1,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("Answer briefly in one paragraph:")[-1].strip()
    return clean_answer(answer)

# -----------------------------
# Build and Launch Gradio UI
# -----------------------------
title = "🩺 Medical Q&A Assistant (BioGPT + LoRA)"
description = "Ask a medical question and get a concise, accurate answer."

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here...", label="Medical Question"),
    outputs=gr.Textbox(label="Answer"),
    title=title,
    description=description,
    examples=[
        ["What are the symptoms of diabetes?"],
        ["How is hypertension treated?"],
        ["What is the difference between a virus and a bacteria?"],
        ["Can you explain the causes of asthma?"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)