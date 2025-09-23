# src/prepare_data.py

import os
import re
import stopwordsiso

from sklearn.feature_extraction.text import TfidfVectorizer

BASE_FOLDER = "/content/drive/MyDrive/medical-text-classifier-clean/data"

# Map SL_data subfolders to main categories
SL_MAPPING = {
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
    "ortodoncja": "Ortodoncja",
    "stomatologia_zachowawcza": "Stomatologia",
    "gastroenterologia_dziecieca": "Gastrologia",
    "ortopedia": "Ortopedia",
    "hematologia": "Hematologia",
    "otolaryngologia": "Otolaryngologia",
}

def load_data(base_folder=BASE_FOLDER):
    texts, labels = [], []

    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)

        # Skip files/folders not meant for training
        if folder in ("not_included", "MedicalTextClassifier_README.txt"):
            continue

        if os.path.isdir(folder_path):
            # Handle SL_data separately
            if folder == "SL_data":
                for sub in os.listdir(folder_path):
                    sub_path = os.path.join(folder_path, sub)
                    if os.path.isdir(sub_path) and sub in SL_MAPPING:
                        for txt_file in os.listdir(sub_path):
                            if txt_file.endswith(".txt"):
                                file_path = os.path.join(sub_path, txt_file)
                                with open(file_path, "r", encoding="utf-8") as f:
                                    texts.append(f.read())
                                    labels.append(SL_MAPPING[sub])
            else:
                # Regular category folders
                for txt_file in os.listdir(folder_path):
                    if txt_file.endswith(".txt"):
                        file_path = os.path.join(folder_path, txt_file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            texts.append(f.read())
                            labels.append(folder)

    return texts, labels


# Preprocessing
polish_stopwords = stopwordsiso.stopwords("pl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\d]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    filtered_words = [w for w in words if w not in polish_stopwords]
    return " ".join(filtered_words)

def preprocess_and_vectorize(texts):
    cleaned = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(
        min_df=1, max_features=500, ngram_range=(2, 2)
    )
    X = vectorizer.fit_transform(cleaned)
    return X, vectorizer
