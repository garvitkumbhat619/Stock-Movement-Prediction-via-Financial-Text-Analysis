from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = "models/bert_tweet_model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API is up 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

    return jsonify({
        "text": text,
        "predicted_label": pred_label,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
