from transformers import BertTokenizer, BertForSequenceClassification, Trainer
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load tokenizer and model
model_path = "models/bert_tweet_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load and clean test dataset
df = pd.read_csv("train.csv")  # or your cleaned file
df = df.dropna(subset=['tweet'])  # safety
df = df.sample(frac=1, random_state=42)  # shuffle

# Split (80% train, 20% test)
split = int(0.8 * len(df))
test_df = df.iloc[split:]

# Tokenize
def tokenize_fn(batch):
    return tokenizer(batch['tweet'], padding='max_length', truncation=True, max_length=64)

from datasets import Dataset
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(tokenize_fn, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Inference with Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Create dataloader
test_loader = DataLoader(test_dataset, batch_size=16)

# Run inference
all_preds = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())

y_pred = np.array(all_preds)
y_test = test_df['label'].values


# Report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BERT")
plt.show()


