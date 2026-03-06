"""
Simple model training script for demonstration
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_simple_model():
    """Train a simple model for demonstration"""
    print("Setting up model training...")

    # Create sample data
    texts = [
        "You're such a loser",
        "Great job!",
        "I hate you",
        "Thanks for your help",
        "Go die",
        "Have a nice day",
        "You're stupid",
        "Good work"
    ]

    labels = [1, 0, 1, 0, 1, 0, 1, 0]

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    # Create datasets
    train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)
    val_dataset = SimpleDataset(val_texts, val_labels, tokenizer)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model directory
    model_dir = Path("models/bert_model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the model (in a real scenario, you would train it)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print(f"Model saved to {model_dir}")

    return model, tokenizer


if __name__ == "__main__":
    train_simple_model()