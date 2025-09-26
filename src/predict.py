# src/predict.py

import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/baseline_lr.pkl")

def load_model():
    """Load trained model and vectorizer"""
    model, vectorizer = joblib.load(MODEL_PATH)
    return model, vectorizer

def predict(text, model, vectorizer, top_k=3):
    """Predict category for a given text and return top_k predictions"""
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    # Sort probabilities descending
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    top_classes = classes[sorted_idx]
    top_probs = probs[sorted_idx]
    
    return list(zip(top_classes, top_probs))

def main():
    model, vectorizer = load_model()
    print("Model loaded. Type a Polish medical text to classify (type 'exit' to quit).")

    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            break
        predictions = predict(text, model, vectorizer)
        print("Top predictions:")
        for cls, prob in predictions:
            print(f"{cls:15} Probability: {prob:.3f}")

if __name__ == "__main__":
    main()
