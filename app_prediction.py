import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Insurance Review Predictor", page_icon="🔮", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #FFF0F3; }
    .stTextArea textarea { border-radius: 12px; font-size: 16px; }
    .stButton button {
        background-color: #4F46E5;
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 12px;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    .stButton button:hover { background-color: #4338CA; }
    .result-card {
        background-color: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .result-title { font-size: 13px; color: #6B7280; font-weight: 600; text-transform: uppercase; }
    .result-value { font-size: 32px; font-weight: 700; color: #111827; margin: 8px 0; }
    .result-sub { font-size: 22px; }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1F2937;
        margin: 24px 0 12px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align:center; padding: 30px 0 10px 0'>
        <h1 style='font-size:2.5rem; color:#4F46E5;'>🔮 Review Predictor</h1>
        <p style='color:#6B7280; font-size:1.1rem;'>
            Enter an insurance review and get instant predictions
        </p>
    </div>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    df = pd.read_csv("topic_dataset.csv")
    df = df.dropna(subset=['note', 'avis_en', 'topic'])
    df['note'] = df['note'].astype(int)

    def map_sentiment(note):
        if note <= 2: return 'Negative'
        elif note == 3: return 'Neutral'
        else: return 'Positive'
    df['sentiment'] = df['note'].apply(map_sentiment)

    star_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    star_pipeline.fit(df['avis_en'], df['note'])

    topic_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    topic_pipeline.fit(df['avis_en'], df['topic'])

    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    sentiment_pipeline.fit(df['avis_en'], df['sentiment'])

    return star_pipeline, topic_pipeline, sentiment_pipeline

with st.spinner("Loading models..."):
    star_model, topic_model, sentiment_model = load_models()

# Input
st.markdown("<div class='section-title'>✍️ Your Review</div>", unsafe_allow_html=True)
review = st.text_area("", height=160,
    placeholder="e.g. The customer service was terrible, I waited 3 months for my reimbursement...")

if st.button("🔮 Analyze Review"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing your review..."):
            star_pred = star_model.predict([review])[0]
            topic_pred = topic_model.predict([review])[0]
            sentiment_pred = sentiment_model.predict([review])[0]
            star_proba = star_model.predict_proba([review])[0]
            topic_proba = topic_model.predict_proba([review])[0]

        sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}
        sentiment_color = {"Positive": "#10B981", "Neutral": "#F59E0B", "Negative": "#EF4444"}
        stars_display = "⭐" * star_pred + "☆" * (5 - star_pred)

        st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class='result-card'>
                    <div class='result-title'>⭐ Star Rating</div>
                    <div class='result-value'>{star_pred} / 5</div>
                    <div class='result-sub'>{stars_display}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class='result-card'>
                    <div class='result-title'>😊 Sentiment</div>
                    <div class='result-value' style='color:{sentiment_color[sentiment_pred]}'>{sentiment_pred}</div>
                    <div class='result-sub'>{sentiment_emoji[sentiment_pred]}</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class='result-card'>
                    <div class='result-title'>🏷️ Topic</div>
                    <div class='result-value' style='font-size:22px'>{topic_pred}</div>
                    <div class='result-sub'>📌</div>
                </div>
            """, unsafe_allow_html=True)

        # Confidence scores
        st.markdown("<div class='section-title'>📈 Confidence Scores</div>", unsafe_allow_html=True)
        col4, col5 = st.columns(2)

        with col4:
            st.markdown("**⭐ Star Rating Probabilities**")
            for i, prob in enumerate(star_proba):
                st.progress(float(prob), text=f"{i+1} ⭐ — {prob:.1%}")

        with col5:
            st.markdown("**🏷️ Topic Probabilities**")
            for topic, prob in zip(topic_model.classes_, topic_proba):
                st.progress(float(prob), text=f"{topic} — {prob:.1%}")