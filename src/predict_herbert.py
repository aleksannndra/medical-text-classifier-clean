from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.functional as F
import joblib
import requests
import io

# -------- CONFIG --------
HF_REPO = "aleksannndra/medical-text-classifier-herbert-PL"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3

# -------- LOAD MODEL & TOKENIZER --------
model = BertForSequenceClassification.from_pretrained(HF_REPO)
tokenizer = BertTokenizer.from_pretrained(HF_REPO)
model.to(DEVICE)
model.eval()

# -------- LOAD LABEL ENCODER --------
LE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/label_encoder.pkl"
r = requests.get(LE_URL)
le = joblib.load(io.BytesIO(r.content))

# -------- PREDICTION FUNCTION --------
def predict(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    top_indices = probs.argsort()[-TOP_K:][::-1]
    results = [(le.classes_[i], float(probs[i])) for i in top_indices]
    return results

# -------- CLI LOOP --------
if __name__ == "__main__":
    print("HerBERT Medical Text Classifier (Polish)")
    print("Type a medical text in Polish to classify. Type 'exit' to exit.")

    while True:
        text = input("\nEnter medical text in Polish: ")
        if text.lower() in ["quit", "exit"]:
            break
        predictions = predict(text)
        print("\nTop predictions:")
        for label, prob in predictions:
            print(f"  {label:<20} {prob:.4f}")
