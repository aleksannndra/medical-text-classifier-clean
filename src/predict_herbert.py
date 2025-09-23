# src/predict_herbert.py

import os
import torch
import joblib
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# -------- CONFIG --------
MODEL_DIR = "/content/drive/MyDrive/medical-text-classifier-clean/models/herbert_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3

# Load model, tokenizer, label encoder
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
model.to(DEVICE)
model.eval()

def predict(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    # Top-k results
    top_indices = probs.argsort()[-TOP_K:][::-1]
    results = [(label_encoder.classes_[i], float(probs[i])) for i in top_indices]
    return results

if __name__ == "__main__":
    while True:
        text = input("\nEnter medical text (or 'quit'): ")
        if text.lower() in ["quit", "exit"]:
            break
        predictions = predict(text)
        print("\nTop predictions:")
        for label, prob in predictions:
            print(f"  {label:<20} {prob:.4f}")

