import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="Insurance RAG", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #FFF0F3; }
    .stButton button {
        background-color: #DC2626;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px;
        border: none;
        width: 100%;
    }
    .chat-user {
        background-color: #4F46E5;
        color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        font-size: 15px;
    }
    .chat-bot {
        background-color: white;
        color: #1F2937;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .source-card {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        border-left: 3px solid #4F46E5;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0'>
        <h1 style='font-size:2.5rem; color:#DC2626;'>🤖 Insurance RAG</h1>
        <p style='color:#6B7280; font-size:1.1rem;'>
            Ask any question about insurance reviews — powered by RAG
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
    return pipeline("text2text-generation", model="t5-small")

df = load_data()
vectorizer, matrix = build_index(df)
qa_model = load_qa_model()

def retrieve_context(query, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results['score'] = scores[top_indices]
    return results[results['score'] > 0]

def generate_answer(query, context_reviews):
    context = " ".join(context_reviews['avis_en'].astype(str).tolist())[:800]
    prompt = f"Based on these insurance reviews: {context} Answer this question: {query}"
    result = qa_model(prompt, max_length=150, min_length=20)
    return result[0]['generated_text']

# Chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
        if 'sources' in msg:
            with st.expander("📚 Sources used"):
                for _, row in msg['sources'].iterrows():
                    st.markdown(f"""
                        <div class='source-card'>
                            <b>{row['assureur']}</b> — ⭐{row['note']} — 🏷️ {row.get('topic', 'N/A')}<br>
                            {str(row['avis_en'])[:200]}...
                        </div>
                    """, unsafe_allow_html=True)

# Input
st.markdown("<br>", unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g. What do customers say about Direct Assurance claims?")

col1, col2 = st.columns([3, 1])
with col2:
    ask_btn = st.button("🤖 Ask", use_container_width=True)
with col1:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if ask_btn and query.strip():
    st.session_state.messages.append({'role': 'user', 'content': query})

    with st.spinner("Searching reviews and generating answer..."):
        context = retrieve_context(query)
        answer = generate_answer(query, context)

    st.session_state.messages.append({
        'role': 'bot',
        'content': answer,
        'sources': context
    })
    st.rerun()