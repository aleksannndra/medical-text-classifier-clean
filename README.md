# Polish Medical Text Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)  
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)  

**Medical Text Classifier** is a Natural Language Processing (NLP) project for classifying **Polish medical texts** into categories such as *Alergie*, *Cukrzyca*, etc.  

It demonstrates both **traditional ML (Logistic Regression + TF-IDF)** and **deep learning (HerBERT, a Polish BERT model)** approaches for medical text classification.  

This project highlights **data collection, preprocessing, model training, and evaluation** — making it a strong showcase for **NLP, ML, and deep learning skills**.  


## Features  

- Text preprocessing with stopword removal, tokenization, and cleaning  
- Baseline classifier using **Logistic Regression**  
- Transformer-based classifier using **HerBERT** (BERT for Polish)  
- Web scraping pipeline for collecting medical texts (`scrape_data.py`)  
- Model persistence (`pickle` for ML, HuggingFace save for BERT)  
- Easy CLI prediction with probabilities  


## Project Structure  
```
medical-text-classifier-clean/
│
├── README.md # This file
├── requirements.txt # Dependencies
├── data/ # example text files
│ ├── Alergie/
│ ├── Cukrzyca/
│ ├── more folders here - see Limitations section below 
│ └── SL_data/ # data from SpeakLeash
├── models/ # Saved models
│ ├── herbert_model/ # not included here due to size - stored on HuggingFace
│ └── baseline_lr.pkl # Logistic Regression model
└── src/ # Scripts
├── prepare_data.py # Data loading & cleaning
├── train_classifier.py # Train Logistic Regression model
├── predict.py # Predict with Logistic Regression
├── train_herbert.py # Train HerBERT model
├── predict_herbert.py # Predict with HerBERT
└── scrape_data.py # Scrape articles from https://www.mp.pl/pacjent/
```

## Repository Notes  

This repository only contains **lightweight data and models** for demonstration purposes.  
The full dataset and large trained models are **not included** due to size constraints and licensing considerations.  

- `data/` → only includes small example subsets (e.g., `Alergie` and `Cukrzyca`).  
  - Original dataset is much larger and cannot be shared directly here.  
  - You can replace the folder with your own text files to retrain models.  

- `models/` → includes only the baseline Logistic Regression model (`baseline_lr.pkl`).  
  - Full fine-tuned **HerBERT model** is **not included here** (2 GB). 
  - The HerBERT model is available on [![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/aleksannndra/medical-text-classifier-herbert-PL) and downloaded from there when running predict_herbert.py

- `logs/`, `wandb/`, and other experiment outputs are ignored via `.gitignore` to keep the repository clean.  


## Data Sources
- **Web scraping:** [https://www.mp.pl/pacjent/](https://www.mp.pl/pacjent/) – Polish medical education portal  
- **Synthetic text generation:** Gemini and ChatGPT — used to balance and diversify categories  
- **SL_data:** Supplementary texts from the [SpeakLeash](https://speakleash.org/) dataset


## Limitations

This project was built primarily as a **learning exercise**, and therefore the dataset is **limited and imperfect**.  

### Categories Included
The following medical specialties were included in the training data:

- **Alergie**  
- **Cukrzyca**  
- **Dermatologia**  
- **Endokrynologia**  
- **Gastrologia**  
- **Ginekologia**  
- **Hematologia**  
- **Kardiologia**  
- **Nefrologia**  
- **Neurologia**  
- **Okulistyka**  
- **Onkologia**  
- **Ortopedia**  
- **Otolaryngologia**  
- **Pediatria**  
- **Psychiatria**  
- **Pulmonologia**  
- **Rehabilitacja**  
- **Reumatologia**  
- **Stomatologia**  

### Important Notes
- **Not all medical specialties were included**. Categories outside the above list will **not be recognized** by the model.  
- Even within these categories, the amount of data was **relatively small** compared to real-world datasets.  
- As a result, the model is **not perfect** — some classes perform very well, while others have lower recall/precision.  
- The main goal of this project was to **learn the full pipeline**: data collection, preprocessing, traditional ML baseline, and fine-tuning a transformer model (HerBERT).  

**In a production setting, much more data would be required for robust performance across all specialties!**

## Technical Stack
- **Language:** Python 3.10  
- **Libraries:** scikit-learn, Transformers (Hugging Face), PyTorch, pandas, NumPy  
- **Model:** `allegro/herbert-base-cased` (fine-tuned)  
- **Feature extraction (baseline):** TF-IDF + Logistic Regression


## Installation  

**Clone repo**
```
git clone https://github.com/aleksannndra/medical-text-classifier-clean.git
cd medical-text-classifier-clean
```

**Create virtual environment**
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**Install dependencies**
```
pip install -r requirements.txt
```

## Using the project

### Terminal/Local machine
<br>
1. Baseline Logistic Regression

```
python src/predict.py
```
*Example:*

Enter text: Pacjent z cukrzycą typu 2   *(text can be much longer)*

Prediction → Cukrzyca (0.92)

<br>
2. HerBERT (BERT for Polish)

```
python src/predict_herbert.py
```

- Download the fine-tuned HerBERT model + tokenizer from HuggingFace
- Download the label encoder
- Allow you to enter text and see top 3 predictions

*Example:*
```
Enter medical text: Pacjent z cukrzycą typu 2
Top predictions:
  Cukrzyca             0.8721
  Endokrynologia       0.0453
  Gastrologia          0.0218
```

## Google Colab
- Provides a ready-to-run environment in the browser
- No installation needed locally
```
!pip install transformers torch datasets requests
!git clone https://github.com/aleksannndra/medical-text-classifier-clean.git
%cd medical-text-classifier-clean
!python src/predict_herbert.py
```

## Model Performance

I evaluated two approaches for **Polish medical text classification**:  
1. A baseline **TF-IDF + Logistic Regression** model.  
2. A fine-tuned **HerBERT (Polish BERT)** model.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|--------|-----------|-----------------|--------------|-----------|--------------------|-----------------|--------------|
| **Baseline (TF-IDF + Logistic Regression)** | 0.39 | 0.55 | 0.38 | 0.42 | 0.55 | 0.39 | 0.43 |
| **HerBERT (fine-tuned)** | 0.88 | 0.89 | 0.88 | 0.88 | 0.89 | 0.88 | 0.88 |

**Key takeaway:**  
- The **baseline model** provides a reference point, capturing basic linguistic patterns but struggling with class imbalance and medical domain nuances.  
- The **HerBERT model** demonstrates strong domain understanding and robust generalization across nearly all medical specialties, achieving an impressive **F1-score of 0.88**.

### 🔍 Per-Class Results — Logistic Regression (Baseline)

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|------------|---------|-----------|----------|
| Alergie            | 0.44       | 0.16    | 0.23      | 45       |
| Cukrzyca           | 0.56       | 0.36    | 0.44      | 67       |
| Dermatologia       | 0.51       | 0.32    | 0.40      | 56       |
| Endokrynologia     | 0.54       | 0.21    | 0.30      | 67       |
| Gastrologia        | 0.81       | 0.57    | 0.67      | 60       |
| Ginekologia        | 0.19       | 0.06    | 0.09      | 54       |
| Hematologia        | 0.65       | 0.35    | 0.45      | 63       |
| Kardiologia        | 0.65       | 0.60    | 0.62      | 82       |
| Nefrologia         | 0.65       | 0.44    | 0.52      | 71       |
| Neurologia         | 0.53       | 0.30    | 0.38      | 71       |
| Okulistyka         | 0.71       | 0.34    | 0.46      | 50       |
| Onkologia          | 0.44       | 0.29    | 0.35      | 68       |
| Ortopedia          | 0.55       | 0.43    | 0.48      | 65       |
| Otolaryngologia    | 0.58       | 0.47    | 0.52      | 66       |
| Pediatria          | 0.60       | 0.39    | 0.47      | 88       |
| Psychiatria        | 0.56       | 0.27    | 0.36      | 52       |
| Pulmonologia       | 0.63       | 0.38    | 0.47      | 69       |
| Rehabilitacja      | 0.12       | 0.88    | 0.21      | 74       |
| Reumatologia       | 0.59       | 0.42    | 0.49      | 69       |
| Stomatologia       | 0.67       | 0.39    | 0.49      | 57       |

| Metric         | Precision | Recall | F1-Score | Support |
|----------------|------------|---------|-----------|----------|
| **Accuracy**        | –          | –       | **0.39**  | 1294     |
| **Macro Avg**       | 0.55       | 0.38    | 0.42      | 1294     |
| **Weighted Avg**    | 0.55       | 0.39    | 0.43      | 1294     |


### 🔍 Per-Class Results — HerBERT (Fine-Tuned)

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Alergie            | 0.89      | 0.88   | 0.88     | 48      |
| Cukrzyca           | 0.86      | 0.91   | 0.88     | 67      |
| Dermatologia       | 0.88      | 0.88   | 0.88     | 56      |
| Endokrynologia     | 0.85      | 0.90   | 0.87     | 67      |
| Gastrologia        | 0.86      | 0.85   | 0.86     | 60      |
| Ginekologia        | 0.85      | 0.88   | 0.86     | 57      |
| Hematologia        | 0.84      | 0.91   | 0.87     | 67      |
| Kardiologia        | 0.89      | 0.89   | 0.89     | 82      |
| Nefrologia         | 0.93      | 0.93   | 0.93     | 71      |
| Neurologia         | 0.88      | 0.86   | 0.87     | 71      |
| Okulistyka         | 0.93      | 0.84   | 0.88     | 50      |
| Onkologia          | 0.89      | 0.83   | 0.86     | 71      |
| Ortopedia          | 0.81      | 0.93   | 0.86     | 68      |
| Otolaryngologia    | 0.97      | 0.92   | 0.95     | 66      |
| Pediatria          | 0.82      | 0.84   | 0.83     | 91      |
| Psychiatria        | 0.96      | 0.87   | 0.91     | 52      |
| Pulmonologia       | 0.93      | 0.90   | 0.91     | 69      |
| Rehabilitacja      | 0.87      | 0.90   | 0.88     | 77      |
| Reumatologia       | 0.89      | 0.84   | 0.87     | 69      |
| Stomatologia       | 0.96      | 0.93   | 0.95     | 57      |

| Metric         | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| **Accuracy**       | –         | –      | 0.88     | 1316    |
| **Macro Avg**      | 0.89      | 0.88   | 0.88     | 1316    |
| **Weighted Avg**   | 0.89      | 0.88   | 0.88     | 1316    |



## License

This project is open-source and available under the MIT License. 


## References

- **HerBERT (Polish BERT model)**: [https://huggingface.co/dkleczek/bert-base-polish-cased-v1](https://huggingface.co/dkleczek/bert-base-polish-cased-v1)  
- **Polish medical dataset sources**:
  - [mp.pl – Pacjent](https://www.mp.pl/pacjent/) (web-scraped articles)
  - SpeakLeash dataset (private mapping for research purposes)
- **Python libraries & tools**:
  - [scikit-learn](https://scikit-learn.org/)
  - [PyTorch](https://pytorch.org/)
  - [HuggingFace Transformers](https://huggingface.co/transformers/)
  - [datasets](https://huggingface.co/docs/datasets/)
  - [HuggingFace Hub](https://huggingface.co/docs/hub/)
  - [stopwordsiso](https://github.com/stopwords-iso/stopwords-iso)
  - [Trafilatura](https://trafilatura.readthedocs.io/en/latest/)
- **Visualization**:
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)































