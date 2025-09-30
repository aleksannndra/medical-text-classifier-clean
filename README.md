# Medical Text Classifier  

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
  - The HerBERT model is available on HuggingFace and downloaded from there when running predict_herbert.py

- `logs/`, `wandb/`, and other experiment outputs are ignored via `.gitignore` to keep the repository clean.  

## Limitations

This project was built primarily as a **learning exercise**, and therefore the dataset is **limited and imperfect**. The data comes from two sources:  

- **[mp.pl](https://www.mp.pl/pacjent/)** → I manually scraped articles across various medical specialties.
- **[SpeakLeash](https://speakleash.org/)** → additional Polish text data, which I mapped to medical categories to expand coverage.  

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

I evaluated two approaches for Polish medical text classification: a baseline TF-IDF + Logistic Regression model, and a fine-tuned HerBERT (Polish BERT) model.

| Model                          | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|--------------------------------|----------|-----------------|--------------|----------|--------------------|-----------------|-------------|
| **Baseline (TF-IDF + Logistic Regression)** | 0.75     | 0.86            | 0.70         | 0.74     | 0.82               | 0.75            | 0.76        |
| **HerBERT (fine-tuned)**       | 0.902    | 0.912           | 0.902        | 0.902    | –                  | –               | –           |

**Key takeaway**:  
- The **baseline model** achieves good performance (75% accuracy, Weighted F1 = 0.76), demonstrating that traditional ML methods can capture meaningful patterns.  
- The **HerBERT model** significantly improves results (Accuracy = 90.2%, F1 = 0.902), showing the advantage of using transformer-based language models for domain-specific classification.


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































