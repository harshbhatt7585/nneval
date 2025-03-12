import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import logging
from scipy.stats import spearmanr
import re
import matplotlib.pyplot as plt  # Added for plotting

# Set up logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_json(text):
    """
    Extracts a JSON object from text using regex.
    
    Args:
        text (str): The input text containing a JSON object.
    
    Returns:
        str: The extracted JSON string, or None if not found.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group()
    return None

# Environment (RAG + LLM)
class RAGEnv:
    def __init__(self, llm_model, judge_model, knowledge_base):
        self.llm_model = llm_model
        self.judge_model = judge_model
        self.knowledge_base = knowledge_base
        self.questions = list(self.knowledge_base.keys())

    def step(self, question):
        """Simulate LLM response and evaluate it with the judge model."""
        try:
            response = self.llm_model(question)['choices'][0]['text'].strip()
        except Exception as e:
            logging.error(f"Error generating response for question: {question}")
            response = ""
        judge_prompt = (
            "<|User|>Evaluate the following answer. Respond strictly in JSON format. "
            "Example format: {\"correct\": true, \"explanation\": \"...\", \"confidence\": 0.9}\n"
            f"Question: {question}\nAnswer: {response}\nResponse in JSON:<|Assistant|>"
        )
        judge_response = self.judge_model(judge_prompt, echo=False, max_tokens=100)['choices'][0]['text'].strip()
        try:
            json_str = extract_json(judge_response)
            judge_data = json.loads(json_str)
            correct = judge_data.get("correct", False)
            confidence = judge_data.get("confidence", 0.0)
            reward = (1 - confidence) if correct else confidence
            logging.info(f"Question: {question}, Response: {response}, Reward: {reward:.4f}")
        except json.JSONDecodeError:
            logging.warning(f"JSON decoding failed for response: {judge_response}")
            reward = 0.0  # Fallback

        except TypeError:
            logging.warning(f"Type error for response: {judge_response}")
            reward = 0.0

        return response, reward

    def sample_question(self):
        """Sample a random question from the knowledge base."""
        return random.choice(self.questions)

# RL Agent
class RLAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RLAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )

    def forward(self, x):
        return self.fc(x)

# Training Process
def train_agent(agent, env, optimizer, encoder, train_questions, num_epochs=10):
    """Train the agent to predict rewards over multiple epochs."""
    losses = []  # To store average loss per epoch
    for epoch in range(num_epochs):
        random.shuffle(train_questions)  # Shuffle questions each epoch
        total_loss = 0.0
        for question in train_questions:
            embedding = torch.tensor(encoder.encode(question), dtype=torch.float32)
            embedding = embedding / embedding.norm()
            response, reward = env.step(question)
            predicted_reward = agent(embedding)
            loss = (predicted_reward - reward) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_questions)
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    return losses

# Evaluation Process
def evaluate_agent(agent, env, encoder, test_questions):
    """Evaluate the agent's ability to rank questions by difficulty."""
    predicted_rewards = []
    actual_rewards = []
    for question in test_questions:
        embedding = torch.tensor(encoder.encode(question), dtype=torch.float32)
        embedding = embedding / embedding.norm()
        predicted_reward = agent(embedding).item()
        _, actual_reward = env.step(question)
        predicted_rewards.append(predicted_reward)
        actual_rewards.append(actual_reward)
        logging.info(f"Test Question: {question}, Predicted Reward: {predicted_reward:.4f}, Actual Reward: {actual_reward:.4f}")
    correlation, p_value = spearmanr(predicted_rewards, actual_rewards)
    logging.info(f"Spearman's Rank Correlation: {correlation:.4f}, p-value: {p_value:.4f}")
    return correlation, predicted_rewards, actual_rewards

# Main Execution
if __name__ == "__main__":
    # Initialize models
    llm_model = Llama(model_path="/Users/harshbhatt/Projects/ai-projects/book-reader/gguf/Llama-3.2-1B-Instruct-f16.gguf",  n_gpu_layers=40, use_mps=True)
    judge_model = Llama(model_path="/Users/harshbhatt/Projects/ai-projects/book-reader/gguf/deepseek-coder-1.3b-instruct.Q8_0.gguf", n_gpu_layers=40, use_mps=True)
    
    try:
        encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Define knowledge base
        knowledge_base = {
            "What is the capital of France?": "Paris",
            "Who wrote 1984?": "George Orwell",
            "What is the speed of light?": "299,792 km/s",
            "Who painted the Mona Lisa?": "Leonardo da Vinci",
            "What is the tallest mountain in the world?": "Mount Everest",
            "Who discovered penicillin?": "Alexander Fleming",
            "What is the boiling point of water in Celsius?": "100",
            "Who was the first president of the United States?": "George Washington",
            "What is the largest planet in our solar system?": "Jupiter",
        }

        # Split into train and test sets
        all_questions = list(knowledge_base.keys())
        random.shuffle(all_questions)
        train_size = int(0.8 * len(all_questions))
        train_questions = all_questions[:train_size]
        test_questions = all_questions[train_size:]

        # Initialize environment and agent
        rag_env = RAGEnv(llm_model, judge_model, knowledge_base)
        agent = RLAgent(input_dim=encoder.get_sentence_embedding_dimension(), hidden_dim=32)
        optimizer = optim.Adam(agent.parameters(), lr=0.001)

        # Train and evaluate
        num_epochs = 1  # Increased for better visualization
        print("Training the agent...")
        losses = train_agent(agent, rag_env, optimizer, encoder, train_questions, num_epochs=num_epochs)
        print("\nEvaluating the agent...")
        correlation, predicted_rewards, actual_rewards = evaluate_agent(agent, rag_env, encoder, test_questions)
    
    finally:
        llm_model.close()
        judge_model.close()

    # Plotting
    # 1. Training Loss Over Epochs
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    logging.info("Training loss plot saved as 'training_loss.png'")

    # 2. Predicted vs Actual Rewards Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_rewards, predicted_rewards, color='blue', label='Data Points')
    plt.xlabel('Actual Rewards')
    plt.ylabel('Predicted Rewards')
    plt.title(f'Predicted vs Actual Rewards\nSpearman Correlation: {correlation:.4f}')
    # Add regression line
    slope, intercept = np.polyfit(actual_rewards, predicted_rewards, 1)
    plt.plot(np.array(actual_rewards), slope * np.array(actual_rewards) + intercept, color='red', label=f'Regression (slope={slope:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('predicted_vs_actual.png')
    plt.show()
    logging.info("Predicted vs Actual rewards plot saved as 'predicted_vs_actual.png'")

