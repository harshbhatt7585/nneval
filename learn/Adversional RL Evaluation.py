import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.distributions import Categorical
from llama_cpp import Llama

# Load Defender LLM (GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Adversarial LLM (Using Local Mistral Model)
adversary_llm = Llama(model_path="/Users/harshbhatt/Projects/ai-projects/book-reader/gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_gpu_layers=40)

# Reward Calculation
def compute_reward(response, original_prompt):
    """ Reward adversary if LLM fails, reward LLM if it handles adversarial input """
    response_tokens = tokenizer.encode(response, return_tensors='pt').to(device)
    prompt_tokens = tokenizer.encode(original_prompt, return_tensors='pt').to(device)
    
    response_len = response_tokens.shape[1]
    prompt_len = prompt_tokens.shape[1]
    
    reward = -abs(response_len - prompt_len)  # Penalize if response is too different
    return reward

# Generate Adversarial Prompt using Local Mistral LLM
def generate_adversarial_prompt(prompt):
    adversarial_prompt = adversary_llm(f"Make this question misleading: {prompt}")
    return adversarial_prompt['choices'][0]['text'].strip()

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    prompt = "What is the capital of France?"
    
    # Generate adversarial prompt dynamically
    adversarial_prompt = generate_adversarial_prompt(prompt)
    
    # LLM Generates a Response
    input_ids = tokenizer.encode(adversarial_prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Compute Reward
    reward = compute_reward(response, prompt)
    reward_tensor = torch.tensor(reward, dtype=torch.float32, requires_grad=True).to(device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Original Prompt: {prompt}")
    print(f"Adversarial Prompt: {adversarial_prompt}")
    print(f"LLM Response: {response}")
    print(f"Reward: {reward:.4f}\n")

print("Training complete.")

# Testing Loop
def test_adversarial_resistance(test_prompts):
    model.eval()
    for prompt in test_prompts:
        adversarial_prompt = generate_adversarial_prompt(prompt)
        input_ids = tokenizer.encode(adversarial_prompt, return_tensors='pt').to(device)
        output = model.generate(input_ids, max_length=50)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Adversarial Prompt: {adversarial_prompt}")
        print(f"LLM Response: {response}\n")
    model.train()

test_prompts = [
    "What is the capital of Italy?",
    "Who discovered gravity?",
    "When did the United States gain independence?"
]

test_adversarial_resistance(test_prompts)