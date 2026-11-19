# News Classification using NLP

Classifies news articles into **World, Sports, Business, Sci/Tech** using the AG News dataset (120,000 training / 7,600 test articles).
Pipeline: EDA → text preprocessing → BoW / TF-IDF features → model training & comparison → prediction system → Flask web app.

## Results (AG News test set, 7,600 articles)

| Model | Features | Accuracy | Weighted F1 |
|-------|----------|---------:|------------:|
| **LinearSVC (selected)** | TF-IDF (1-2 grams) | **92.36%** | **0.923** |
| Logistic Regression | TF-IDF | 92.00% | 0.920 |
| Multinomial Naive Bayes | TF-IDF | 90.76% | 0.907 |
| Logistic Regression | BoW | 90.24% | 0.902 |
| Naive Bayes | BoW | 90.18% | 0.902 |
| Random Forest | TF-IDF (chi2 top 20k) | 89.89% | 0.899 |
| Gradient Boosting (HistGB) | TF-IDF + SVD(200) | 89.04% | 0.890 |

LinearSVC was chosen for the best F1, the fastest training/prediction, and a simple linear model. It is wrapped in probability calibration so the app can show confidence scores. Charts are in `reports/`.

## Project layout

| Path | Purpose |
|------|---------|
| `01_EDA.ipynb` | Exploratory data analysis |
| `02_Text Preprocessing.ipynb` | First preprocessing experiments |
| `03_Feature_Extraction.ipynb` | Cleans text, fits BoW and TF-IDF vectorizers |
| `04_Model_Training_Evaluation.ipynb` | Trains, tunes and evaluates 5 model families |
| `05_Model_Comparison_Prediction.ipynb` | Compares models, selects the best, saves artifacts, tests `predict()` |
| `src/preprocess.py` | Shared `clean_text()` used in training and inference |
| `src/predict.py` | `predict(text)` → category, confidence, per-class probabilities |
| `app/` | Flask app (`app.py`, HTML/CSS/JS frontend) |
| `models/` | Deployable artifacts: `best_model.joblib`, `tfidf_vectorizer.joblib`, `label_encoder.joblib`, `model_info.json` |
| `reports/` | Metrics (`results.json`) and charts |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the web app

```bash
python app/app.py                 # http://localhost:5000  (set PORT to change)
# or, production-style:
gunicorn -w 2 -b 0.0.0.0:5000 app.app:app
```

API: `POST /api/predict` with JSON `{"text": "..."}` returns `{"category", "confidence", "probabilities"}`; `GET /api/model` returns model details; `GET /api/health` for health checks.

## Use from Python / command line

```python
from src.predict import predict
predict("Oil prices fell sharply as investors weighed new supply data.")
```

```bash
python -m src.predict "Manchester United edged Chelsea 2-1 in the league"
```

## Retrain from scratch

Run notebooks `03` → `04` → `05` in order from the repository root (`jupyter nbconvert --to notebook --execute --inplace <notebook>`). Notebook 04 takes about 10 minutes on 2 CPU cores, mostly Random Forest. Intermediate files (`data/processed/`, `models/features/`, non-deployed models) are git-ignored.

## Notes

- Preprocessing: lowercase, remove URLs/HTML/special characters, stopwords (scikit-learn list), Porter stemming. Raw AG News contains literal backslashes as line breaks; they are converted to spaces. `train_cleaned.csv` / `test_cleaned.csv` come from the earlier notebook (02) and are not used by the later notebooks.
- Word2Vec embeddings were not used; BoW and TF-IDF cover the feature-extraction comparison.
- Only AG News is used; `news_classification_dataset.csv` mentioned in the original plan is not part of this repository.
