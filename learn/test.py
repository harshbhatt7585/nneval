import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset("stanfordnlp/SHP", split="train[:1%]")

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)

# Tokenization function
def preprocess(examples):
    return tokenizer(examples["human_ref_A"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess, batched=True)

# Convert only numerical tensors to float32
def convert_dtype(batch):
    for key in batch:
        if isinstance(batch[key], list) and isinstance(batch[key][0], (int, float)):  # Only convert numerical fields
            batch[key] = torch.tensor(batch[key], dtype=torch.float32)
    return batch

tokenized_dataset = tokenized_dataset.map(convert_dtype, batched=True)

# Reduce dataset size
train_dataset = tokenized_dataset.shuffle(seed=42).select(range(min(100, len(tokenized_dataset))))

# Training arguments
training_args = TrainingArguments(
    output_dir="./reward_model",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
trainer.train()
