import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.set_page_config(page_title="Prediction Explainer", page_icon="🔍", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #FFF0F3; }
    .stButton button {
        background-color: #7C3AED;
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 12px;
        width: 100%;
        border: none;
    }
    .result-card {
        background-color: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; padding: 30px 0 10px 0'>
        <h1 style='font-size:2.5rem; color:#7C3AED;'>🔍 Prediction Explainer</h1>
        <p style='color:#6B7280; font-size:1.1rem;'>
            Understand why the model made its prediction
        </p>
    </div>
""", unsafe_allow_html=True)

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

    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    sentiment_pipeline.fit(df['avis_en'], df['sentiment'])

    topic_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    topic_pipeline.fit(df['avis_en'], df['topic'])

    return star_pipeline, sentiment_pipeline, topic_pipeline

with st.spinner("Loading models..."):
    star_model, sentiment_model, topic_model = load_models()

def get_top_words(pipeline, text, top_n=10):
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    vec = tfidf.transform([text])
    pred_class_idx = clf.predict(vec)[0]
    
    if hasattr(clf, 'classes_'):
        class_idx = list(clf.classes_).index(pred_class_idx)
    else:
        class_idx = pred_class_idx - 1
    
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[class_idx] if len(clf.coef_) > 1 else clf.coef_[0]
    
    vec_array = vec.toarray()[0]
    word_scores = {feature_names[i]: coefs[i] * vec_array[i] 
                   for i in range(len(feature_names)) if vec_array[i] > 0}
    
    sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return sorted_words, pred_class_idx

# Input
st.markdown("<div style='font-size:18px; font-weight:700; margin: 24px 0 12px 0'>✍️ Your Review</div>", unsafe_allow_html=True)
review = st.text_area("", height=160,
    placeholder="e.g. I waited 3 months for my reimbursement, terrible customer service...")

if st.button("🔍 Explain Prediction"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            star_pred = star_model.predict([review])[0]
            sentiment_pred = sentiment_model.predict([review])[0]
            topic_pred = topic_model.predict([review])[0]

            star_words, _ = get_top_words(star_model, review)
            sentiment_words, _ = get_top_words(sentiment_model, review)
            topic_words, _ = get_top_words(topic_model, review)

        sentiment_color = {"Positive": "#10B981", "Neutral": "#F59E0B", "Negative": "#EF4444"}

        # Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class='result-card'>
                    <div style='font-size:13px; color:#6B7280; font-weight:600'>⭐ STAR RATING</div>
                    <div style='font-size:32px; font-weight:700'>{star_pred} / 5</div>
                    <div>{"⭐" * star_pred}{"☆" * (5-star_pred)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class='result-card'>
                    <div style='font-size:13px; color:#6B7280; font-weight:600'>😊 SENTIMENT</div>
                    <div style='font-size:32px; font-weight:700; color:{sentiment_color[sentiment_pred]}'>{sentiment_pred}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class='result-card'>
                    <div style='font-size:13px; color:#6B7280; font-weight:600'>🏷️ TOPIC</div>
                    <div style='font-size:22px; font-weight:700'>{topic_pred}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Explanations
        st.markdown("### 🧠 Why did the model predict this?")

        def plot_explanation(words, title, color):
            if not words:
                return
            labels = [w[0] for w in words[:8]]
            scores = [w[1] for w in words[:8]]
            colors = ['#10B981' if s > 0 else '#EF4444' for s in scores]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor='white')
            ax.set_title(title, fontweight='bold', fontsize=13)
            ax.set_xlabel("Word Contribution Score")
            ax.axvline(0, color='black', linewidth=0.8)
            fig.patch.set_facecolor('#FFF0F3')
            ax.set_facecolor('#FFF0F3')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        tab1, tab2, tab3 = st.tabs(["⭐ Star Rating", "😊 Sentiment", "🏷️ Topic"])

        with tab1:
            st.markdown(f"The model predicted **{star_pred} stars** because of these words:")
            plot_explanation(star_words, f"Word contributions → {star_pred} stars", "#4F46E5")

        with tab2:
            st.markdown(f"The model detected **{sentiment_pred}** sentiment because of these words:")
            plot_explanation(sentiment_words, f"Word contributions → {sentiment_pred}", sentiment_color[sentiment_pred])

        with tab3:
            st.markdown(f"The model classified this as **{topic_pred}** because of these words:")
            plot_explanation(topic_words, f"Word contributions → {topic_pred}", "#7C3AED")