#  Medical Text Classifier

A simple machine learning project that predicts the **branch of medicine** a given text belongs to.  
It combines **scraped Polish medical articles** (`mp.pl`) with **exam datasets** from [SpeakLeash](https://huggingface.co/speakleash).

---

## Features
- Scrapes Polish medical articles from [mp.pl/pacjent](https://www.mp.pl/pacjent/).
- Loads labeled exam data from `speakleash/PES-2018-2022`.
- Preprocesses text: lowercasing, cleaning, removing Polish stopwords.
- Vectorizes with **TF-IDF (bigrams)**.
- Trains a **Logistic Regression classifier** (plus experiments with NB, SVM, RF).
- Achieves ~83% accuracy on test data.

---

##  Project Structure
```
medical-text-classifier/
├── data/ # Raw + processed text data
├── notebooks/ # EDA and experiments
├── src/ # Source code
│ ├── scrape_data.py # Scrape mp.pl
│ ├── prepare_data.py # Clean & preprocess
│ ├── train_classifier.py # Train & evaluate model
│ └── utils.py # Helper functions
├── models/ # Trained models (optional)
├── requirements.txt # Dependencies
└── README.md # This file
```
## Quickstart

### 1. Clone repo
```
git clone https://github.com/<your-username>/medical-text-classifier.git
cd medical-text-classifier
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Scrape data
```
python src/scrape_data.py
```
### 4. Prepare data
```
python src/prepare_data.py
```
### 5. Train model
```
python src/train_classifier.py
```
## Results
Baseline Logistic Regression:

Accuracy: 0.83

Strong performance on categories like Alergie, Onkologia, Psychiatria, Stomatologia.

Weaker performance for underrepresented categories (e.g., Ginekologia, Nefrologia).

## License
MIT License.

