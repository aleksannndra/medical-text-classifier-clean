# src/train_classifier.py

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from prepare_data import load_data, clean_text

# --- 1. Load and preprocess data ---
BASE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

texts, labels = load_data(BASE_FOLDER)

# Clean texts
texts = [clean_text(t) for t in texts]

# --- 2. Split into train/test ---
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- 3. Vectorize ---
vectorizer = TfidfVectorizer(min_df=1, max_features=500, ngram_range=(2, 2))
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# --- 4. Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 5. Save model + vectorizer together ---
MODEL_PATH = "models/baseline_lr.pkl"
os.makedirs("models", exist_ok=True)
joblib.dump((model, vectorizer), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- 6. Evaluate on test set ---
y_pred = model.predict(X_test)
print("\n=== Test Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
df_cm = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()
