import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Force use of CPU
device = torch.device("cpu")

# Load model + tokenizer
MODEL_PATH = "models/bert_tweet_model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Streamlit UI
st.title("📉 Financial Tweet Sentiment Classifier")

user_input = st.text_area("Enter a financial tweet:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # move to CPU

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred_label = torch.argmax(probs).item()
            confidence = torch.max(probs).item()

        label_name = "Positive" if pred_label == 1 else "Negative"
        st.success(f"**Prediction:** {label_name}")
        st.write(f"**Confidence:** {confidence:.4f}")
