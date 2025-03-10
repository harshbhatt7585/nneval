import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter

# Load Pretrained GPT-2 Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Intrinsic Curiosity Module (ICM)
class ICM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ICM, self).__init__()
        self.feature_extractor = nn.Linear(input_dim, hidden_dim)
        self.forward_model = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, state, next_state):
        state_feat = F.relu(self.feature_extractor(state))
        next_state_feat = F.relu(self.feature_extractor(next_state))
        pred_next_state = self.forward_model(state_feat)
        intrinsic_reward = F.mse_loss(pred_next_state, next_state_feat, reduction='none').mean(dim=-1)
        return intrinsic_reward

# Policy Network for PPO
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

# Initialize Components
icm = ICM(input_dim=768, hidden_dim=768).to(device)
policy_net = PolicyNetwork(input_dim=768, hidden_dim=768, output_dim=1).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(icm.parameters()) + list(policy_net.parameters()), lr=5e-5)

# Generate Diverse Responses
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Compute Diversity Reward
def compute_diversity_reward(outputs):
    embeddings = [model.transformer.wte(torch.tensor(tokenizer.encode(out)).to(device)).mean(dim=0).detach().cpu().numpy() for out in outputs]
    similarity_matrix = cosine_similarity(embeddings)
    avg_similarity = np.mean(similarity_matrix)
    self_bleu = np.mean([sentence_bleu([outputs[:i] + outputs[i+1:]], out) for i, out in enumerate(outputs)])
    return 1 - (avg_similarity + self_bleu) / 2

# Compute Distinct-n Metrics
def distinct_n(outputs, n=2):
    ngrams = [tuple(outputs[i:i+n]) for i in range(len(outputs)-n+1)]
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams) if len(ngrams) > 0 else 0

# PPO Training Loop
num_epochs = 5
clip_epsilon = 0.2
for epoch in range(num_epochs):
    prompt = "Once upon a time"
    outputs = [generate_text(prompt) for _ in range(5)]
    diversity_reward = compute_diversity_reward(outputs)
    
    # Compute ICM reward
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs_states = model.transformer(input_ids).last_hidden_state.mean(dim=1)
    next_states = outputs_states.clone().detach() + torch.randn_like(outputs_states) * 0.1
    intrinsic_reward = icm(outputs_states, next_states).mean()
    
    total_reward = diversity_reward + intrinsic_reward.item()
    
    # Compute PPO Loss
    policy_old = policy_net(outputs_states.detach())
    policy_new = policy_net(outputs_states)
    ratio = torch.exp(policy_new - policy_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    ppo_loss = -torch.min(ratio * total_reward, clipped_ratio * total_reward).mean()
    
    # Backpropagation
    optimizer.zero_grad()
    ppo_loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Diversity Reward: {diversity_reward:.4f}, Intrinsic Reward: {intrinsic_reward.item():.4f}, PPO Loss: {ppo_loss.item():.4f}")

# Evaluation
def evaluate():
    prompt = "Once upon a time"
    outputs = [generate_text(prompt) for _ in range(10)]
    self_bleu = np.mean([sentence_bleu([outputs[:i] + outputs[i+1:]], out) for i, out in enumerate(outputs)])
    distinct_1 = distinct_n(' '.join(outputs).split(), n=1)
    distinct_2 = distinct_n(' '.join(outputs).split(), n=2)
    
    print("\nEvaluation Results:")
    print(f"Self-BLEU: {self_bleu:.4f} (Lower is better)")
    print(f"Distinct-1: {distinct_1:.4f} (Higher is better)")
    print(f"Distinct-2: {distinct_2:.4f} (Higher is better)")

# Run evaluation after training
evaluate()

print("Training and evaluation complete.")