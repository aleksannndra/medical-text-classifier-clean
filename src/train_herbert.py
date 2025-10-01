# src/train_herbert.py

import os
import re
import torch
import stopwordsiso
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from google.colab import drive
drive.mount('/content/drive')


# --------- CONFIG ---------
DATA_FOLDER = "/content/drive/MyDrive/medical-text-classifier-clean/data" # adjust if needed
MODEL_NAME = "dkleczek/bert-base-polish-cased-v1"
SAVE_DIR = "/content/drive/MyDrive/medical-text-classifier-clean/models/herbert_model"
MAX_LENGTH = 500
BATCH_SIZE = 8
EPOCHS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- DATA PREPARATION ---------
polish_stopwords = stopwordsiso.stopwords("pl")


def load_data(base_folder=DATA_FOLDER):
    texts, labels = [], []
    # mapping SL_data subfolders
    sl_mapping = {
        "alergologia": "Alergie",
        "kardiochirurgia": "Kardiologia",
        "pediatria": "Pediatria",
        "chirurgia_onkologiczna": "Onkologia",
        "kardiologia": "Kardiologia",
        "poloznictwo_i_ginekologia": "Ginekologia",
        "chirurgia_stomatologiczna": "Stomatologia",
        "nefrologia": "Nefrologia",
        "protetyka_stomatologiczna": "Stomatologia",
        "choroby_pluc": "Pulmonologia",
        "neurochirurgia": "Neurologia",
        "psychiatria": "Psychiatria",
        "choroby_pluc_dzieci": "Pulmonologia",
        "neurologia": "Neurologia",
        "psychiatria_dzieci_i_mlodziezy": "Psychiatria",
        "dermatologia_i_wenerologia": "Dermatologia",
        "neurologia_dziecieca": "Neurologia",
        "rehabilitacja_medyczna": "Rehabilitacja",
        "diabetologia": "Cukrzyca",
        "okulistyka": "Okulistyka",
        "reumatologia": "Reumatologia",
        "endokrynologia": "Endokrynologia",
        "onkologia_kliniczna": "Onkologia",
        "stomatologia_dziecieca": "Stomatologia",
        "gastroenterologia": "Gastrologia",
        "stomatologia_zachowawcza": "Stomatologia",
        "gastroenterologia_dziecieca": "Gastrologia",
        "ortopedia": "Ortopedia",
        "hematologia": "Hematologia",
        "otolaryngologia": "Otolaryngologia"
    }

    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            if folder == "not_included":
                continue
            if folder == "SL_data":
                for sub in os.listdir(folder_path):
                    sub_path = os.path.join(folder_path, sub)
                    if os.path.isdir(sub_path):
                        label = sl_mapping.get(sub)
                        if not label:
                            continue
                        for txt_file in os.listdir(sub_path):
                            if txt_file.endswith(".txt"):
                                with open(os.path.join(sub_path, txt_file), "r", encoding="utf-8") as f:
                                    text = f.read()
                                    texts.append(text)
                                    labels.append(label)
                continue

            # normal folders
            for txt_file in os.listdir(folder_path):
                if txt_file.endswith(".txt") and txt_file != "MedicalTextClassifier_README.txt":
                    with open(os.path.join(folder_path, txt_file), "r", encoding="utf-8") as f:
                        text = f.read()
                        texts.append(text)
                        labels.append(folder)
    return texts, labels

# Load data
texts, labels = load_data()
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
)

train_dataset = Dataset.from_dict({"text": X_train, "labels": y_train})
test_dataset = Dataset.from_dict({"text": X_test, "labels": y_test})
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# --------- TOKENIZATION ---------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# --------- MODEL ---------
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE)

# --------- TRAINER ---------
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

os.environ["WANDB_DISABLED"] = "true"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --------- TRAIN ---------
trainer.train()

# Save model + label encoder
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
import joblib
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))

print("Training completed!")
