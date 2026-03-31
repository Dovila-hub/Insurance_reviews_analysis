#  Insurance Review Analytics — NLP Project

A complete NLP pipeline for analyzing 34,000+ French insurance customer reviews, 
built as part of the ESILV NLP 2026 supervised learning project.

##  Project Overview

This project applies a full range of NLP techniques to a dataset of insurance 
customer reviews from 56 French insurers. The goal is to predict star ratings, 
detect sentiment, identify topics, and deliver insights through interactive 
Streamlit applications.

**Dataset:** 34,428 reviews across 6 topics — Pricing, Claims Processing, 
Customer Service, Coverage, Enrollment, Cancellation.

---

##  Project Structure
```
nlp_project/
├── 1_DataCleaning.ipynb          # Merging, cleaning, spelling correction, n-grams
├── 2_Summary_transl.ipynb        # Translation, summarization with T5
├── 3_Topmodeling.ipynb           # Data visualization + LDA topic modeling
├── 4_embeddings.ipynb            # Word2Vec, GloVe, t-SNE, semantic search
├── 5_Supervised_learning.ipynb   # All ML/DL models + results interpretation
├── app.py                        # Main Streamlit home page
├── pages/
│   ├── 1_Prediction.py           # Star rating + sentiment + topic prediction
│   ├── 2_Summary.py              # Insurer performance analytics
│   ├── 3_Explanation.py          # Word-level prediction explanations
│   ├── 4_Search.py               # Semantic search across 34K reviews
│   ├── 5_RAG.py                  # Retrieval Augmented Generation
│   └── 6_QA.py                   # Question Answering from reviews
├── cleaned_dataset.csv           # Cleaned and corrected dataset
├── enriched_dataset.csv          # Dataset with translations and summaries
└── topic_dataset.csv             # Dataset with LDA topic labels
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Dovila-hub/Insurance_reviews_analysis.git
cd Insurance_reviews_analysis
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install pandas numpy matplotlib wordcloud nltk spacy scikit-learn gensim \
pyspellchecker deep-translator tqdm openpyxl seaborn torch transformers \
sentencepiece streamlit
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

### 4. Launch the Streamlit app
```bash
python -m streamlit run app.py
```

---

##  Results Summary

| Model | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 52% |
| Basic Embedding Model | 51% |
| Pre-trained Word2Vec | 51% |
| CNN | 52% |
| CamemBERT (BERT) | **55%** |
| Sentiment (TF-IDF) | **80%** |

---

##  Tech Stack

- **NLP:** NLTK, spaCy, HuggingFace Transformers, Gensim
- **ML/DL:** Scikit-learn, PyTorch, CamemBERT, RoBERTa
- **Visualization:** Matplotlib, Seaborn, WordCloud, Tensorboard
- **Apps:** Streamlit
- **Data:** Pandas, NumPy

---

##  Key Findings

- **Pricing** is the most discussed topic (13,839 reviews) and the most positive (87%)
- **Cancellation** is the most negative topic (60% negative reviews)
- **CamemBERT** outperforms all models for star rating prediction
- **TF-IDF** surprisingly outperforms pre-trained RoBERTa for sentiment analysis
- Average rating across all insurers: **2.85 / 5**

---

## 👤 Author

**Dovila Longmis** — ESILV NLP 2026
