import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="Insurance Q&A", page_icon="💬", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #FFF0F3; }
    .stButton button {
        background-color: #0EA5E9;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px;
        border: none;
        width: 100%;
    }
    .answer-card {
        background-color: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #0EA5E9;
        margin: 16px 0;
    }
    .source-card {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        border-left: 3px solid #0EA5E9;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0'>
        <h1 style='font-size:2.5rem; color:#0EA5E9;'>💬 Insurance Q&A</h1>
        <p style='color:#6B7280; font-size:1.1rem;'>
            Ask specific questions about insurers and get answers from real reviews
        </p>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("topic_dataset.csv")
    df = df.dropna(subset=['avis_en'])
    df['note'] = df['note'].fillna(0).astype(int)
    return df

@st.cache_resource
def build_index(df):
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    matrix = vectorizer.fit_transform(df['avis_en'].astype(str))
    return vectorizer, matrix

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

df = load_data()
vectorizer, matrix = build_index(df)

with st.spinner("Loading QA model..."):
    qa_model = load_qa_model()

st.success("✅ QA model loaded!")

# Suggested questions
st.markdown("### 💡 Suggested Questions")
suggestions = [
    "What do customers say about reimbursement delays?",
    "How is the customer service at Direct Assurance?",
    "What are the main complaints about cancellation?",
    "Which insurer has the best pricing?"
]

cols = st.columns(2)
for i, suggestion in enumerate(suggestions):
    with cols[i % 2]:
        if st.button(suggestion, use_container_width=True):
            st.session_state['question'] = suggestion

# Input
question = st.text_input("❓ Your Question",
    value=st.session_state.get('question', ''),
    placeholder="e.g. What do customers think about claim processing?")

# Insurer filter
selected_insurer = st.selectbox("🏢 Filter by Insurer (optional)",
    ["All"] + sorted(df['assureur'].unique().tolist()))

if st.button("💬 Get Answer", use_container_width=True):
    if question.strip() == "":
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("Finding answer..."):
            # Filter
            filtered = df.copy()
            if selected_insurer != "All":
                filtered = filtered[filtered['assureur'] == selected_insurer]

            # Retrieve relevant reviews
            sub_matrix = matrix[filtered.index]
            query_vec = vectorizer.transform([question])
            scores = cosine_similarity(query_vec, sub_matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:5]
            top_reviews = filtered.iloc[top_indices]

            # Build context
            context = " ".join(top_reviews['avis_en'].astype(str).tolist())[:1000]

            # QA
            result = qa_model(question=question, context=context)
            answer = result['answer']
            confidence = result['score']

        # Display answer
        st.markdown(f"""
            <div class='answer-card'>
                <div style='font-size:13px; color:#6B7280; font-weight:600; margin-bottom:8px'>
                    💬 ANSWER — Confidence: {confidence:.1%}
                </div>
                <div style='font-size:20px; font-weight:700; color:#1F2937'>{answer}</div>
            </div>
        """, unsafe_allow_html=True)

        # Sources
        st.markdown("### 📚 Sources")
        for _, row in top_reviews.iterrows():
            st.markdown(f"""
                <div class='source-card'>
                    <b>{row['assureur']}</b> — ⭐{row['note']} — 🏷️ {row.get('topic', 'N/A')}<br>
                    {str(row['avis_en'])[:250]}...
                </div>
            """, unsafe_allow_html=True)