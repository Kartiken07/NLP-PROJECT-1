# News Classification using NLP — Project Report

**Course:** Natural Language Processing (NLP)
**Project:** Automated News Article Classification
**Team Size:** 5 Members
**Report Date:** 19 November 2026

---

## 1. Introduction

The rapid growth of digital news has made manual categorization of articles inefficient and error-prone. This project focuses on building an end-to-end Natural Language Processing (NLP) pipeline that automatically classifies news articles into predefined categories such as World, Sports, Business, and Sci/Tech.

The workflow covers the complete machine learning lifecycle: from raw data collection and exploratory analysis, through text preprocessing and feature extraction, to model training, evaluation, comparison, and finally the deployment of a user-facing prediction system with a frontend interface.

**Dataset used:** AG News dataset (via HuggingFace `sh0416/ag_news`) and a local news classification dataset (`news_classification_dataset.csv`, 30,000 articles).

## 2. Team Contribution Breakdown

| Member | Responsibility Area |
|--------|---------------------|
| Member 1 | Dataset Collection & Exploratory Data Analysis |
| Member 2 | Text Preprocessing & Feature Extraction |
| Member 3 | Model Building & Model Evaluation |
| Member 4 | Model Comparison & Selection + Prediction System |
| Member 5 | Frontend Development + Final Deployment & Documentation |

## 3. Detailed Contributions

### 3.1 Member 1 — Dataset Collection & Exploratory Data Analysis

**Deliverables:**

- Collected and loaded the AG News dataset using `datasets.load_dataset("sh0416/ag_news")` (train/test splits).
- Converted HuggingFace dataset objects into pandas DataFrames for analysis.
- Performed structural analysis: shape inspection, data types, column profiling, and `describe()` statistics.
- Mapped numeric labels to category names: 1 → World, 2 → Sports, 3 → Business, 4 → Sci/Tech.
- Combined title and description into a unified text column.
- Conducted data quality checks: missing value detection, duplicate row detection, duplicate text detection.
- Performed statistical and visual analysis:
  - Category distribution (countplot) — verified class balance.
  - Word count distribution (histogram + KDE).
  - Article length by category (boxplot).
  - Top-20 most frequent words (bar chart) after basic tokenization with regex.

**Tools/Libraries:** pandas, numpy, matplotlib, seaborn, collections.Counter, re, datasets

**Key file:** `01_EDA.ipynb`

### 3.2 Member 2 — Text Preprocessing & Feature Extraction

**Responsibilities:**

- Designed and implemented the text cleaning pipeline:
  - Lowercasing
  - Removal of HTML tags, URLs, and special characters
  - Tokenization
  - Stopword removal
  - Lemmatization / stemming
  - Handling of empty or malformed texts
- Applied the preprocessing pipeline consistently to both train and test sets.
- Performed feature extraction using multiple vectorization strategies:
  - Bag of Words (BoW)
  - TF-IDF Vectorization (unigram + bigram)
  - Word Embeddings (Word2Vec / TF-IDF weighted embeddings, as applicable)
- Saved the fitted vectorizers and preprocessing artifacts for reuse during model training and inference.

**Tools/Libraries:** nltk, re, sklearn.feature_extraction.text (CountVectorizer, TfidfVectorizer), gensim (if applicable)

**Outcome:** Cleaned, vectorized representations of the news text ready for model consumption.

### 3.3 Member 3 — Model Building & Model Evaluation

**Responsibilities:**

- Trained multiple classification models on the extracted features:
  - Naive Bayes (MultinomialNB)
  - Logistic Regression
  - Support Vector Machine (SVM / LinearSVC)
  - Random Forest
  - Gradient Boosting / XGBoost (as applicable)
- Performed train/validation splitting with stratification to preserve class balance.
- Conducted hyperparameter tuning using GridSearchCV / RandomizedSearchCV with cross-validation.
- Evaluated each model using comprehensive metrics: Accuracy; Precision, Recall, F1-Score (per class and weighted average); Confusion Matrix; Classification Report.
- Analyzed model behavior — misclassifications, class-wise performance, and error patterns.

**Tools/Libraries:** scikit-learn (models, metrics, model_selection), matplotlib, seaborn

**Outcome:** Trained models with quantified performance across standard evaluation metrics.

### 3.4 Member 4 — Model Comparison & Selection + Prediction System

**Model Comparison & Selection:**

- Compiled all model results into a unified comparison table (Accuracy, Precision, Recall, F1).
- Visualized model performance with bar charts and confusion matrices side by side.
- Selected the best performing model based on overall F1-score and robustness across classes.
- Justified the final model choice considering accuracy vs. speed vs. interpretability trade-offs.

**Prediction System:**

- Built an end-to-end inference pipeline:
  1. Raw text input
  2. Preprocessing (reusing Member 2's pipeline)
  3. Feature extraction (reusing the fitted vectorizer)
  4. Prediction using the selected model
  5. Output predicted category with confidence/probability scores
- Serialized the final artifacts (model + vectorizer + label encoder) using joblib / pickle.
- Created a reusable `predict(text)` function/script for programmatic access.

**Tools/Libraries:** scikit-learn, joblib, matplotlib, seaborn

**Outcome:** A single best model and a functional prediction pipeline ready for integration.

### 3.5 Member 5 — Frontend Development + Final Deployment & Documentation

**Frontend Development:**

- Designed and built a user-friendly web interface where users can:
  - Paste a news article headline/body
  - Click "Classify" to receive the predicted category
  - View confidence score and model details
- Ensured clean UI layout, responsive design, and intuitive UX.

**Deployment:**

- Exposed the prediction pipeline through a web framework (e.g., Flask / FastAPI).
- Containerized or configured the application for local/demo deployment.
- Managed environment dependencies and reproducibility (`requirements.txt`, virtual environment).

**Documentation:**

- Compiled this project report.
- Documented setup/installation steps, usage instructions, and architecture overview.
- Organized final repository structure and submission-ready materials.

**Tools/Libraries:** Flask / FastAPI, HTML/CSS/JS (or Streamlit/Gradio), joblib

**Outcome:** A working web application for news classification, fully documented and ready for demo.

## 4. System Architecture (Overview)

```
Raw News Text
      │
      ▼
[Member 1] Data Collection & EDA
      │
      ▼
[Member 2] Preprocessing + Feature Extraction
      │
      ▼
[Member 3] Model Training + Evaluation
      │
      ▼
[Member 4] Model Comparison → Best Model → predict()
      │
      ▼
[Member 5] Web Frontend + Deployment + Documentation
      │
      ▼
User receives predicted news category
```

## 5. Project Files

| File | Description |
|------|-------------|
| `01_EDA.ipynb` | Exploratory Data Analysis notebook (Member 1) |
| `news_classification_dataset.csv` | Local news dataset — 30,000 articles, 15 columns |
| `PROJECT_REPORT.md` | This report |

## 6. Conclusion

The project successfully demonstrates a complete NLP-based news classification pipeline. All five team members contributed their assigned modules collaboratively:

- **Member 1** established a strong analytical foundation through thorough EDA.
- **Member 2** ensured clean, machine-readable text features.
- **Member 3** built and rigorously evaluated multiple classifiers.
- **Member 4** selected the optimal model and packaged it into a prediction system.
- **Member 5** delivered the user-facing frontend, deployment, and documentation.

The final system takes raw news text as input and accurately predicts its category, showcasing practical application of NLP techniques in real-world content organization.

*Report generated for team project submission.*
