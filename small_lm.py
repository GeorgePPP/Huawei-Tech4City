import transformers
from transformers import AutoTokenizer
import torch

model_name = "M4-ai/TinyMistral-6x248M-Instruct"

# Load tokenizer and model pipeline
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)

def generate_text(prompt, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.95):
    outputs = pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p)
    return outputs[0]["generated_text"]

def apply_chat_template(messages):
    return pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
