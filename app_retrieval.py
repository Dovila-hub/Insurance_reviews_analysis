import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Review Search", page_icon="🔎", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FFF0F3; }
    .stButton button {
        background-color: #059669;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px;
        border: none;
    }
    .review-card {
        background-color: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #059669;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align:center; padding: 30px 0 10px 0'>
        <h1 style='font-size:2.5rem; color:#059669;'>🔎 Review Search</h1>
        <p style='color:#6B7280; font-size:1.1rem;'>
            Search through 34,000+ insurance reviews using keywords or semantic queries
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
def build_search_index(df):
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(df['avis_en'].astype(str))
    return vectorizer, tfidf_matrix

df = load_data()
vectorizer, tfidf_matrix = build_search_index(df)

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_insurer = st.sidebar.selectbox("Insurer", ["All"] + sorted(df['assureur'].unique().tolist()))
selected_topic = st.sidebar.selectbox("Topic", ["All"] + sorted(df['topic'].dropna().unique().tolist()))
selected_sentiment = st.sidebar.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
min_rating, max_rating = st.sidebar.slider("Rating Range", 1, 5, (1, 5))
top_k = st.sidebar.slider("Number of results", 5, 20, 10)

# Search
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("", placeholder="e.g. slow reimbursement, excellent customer service...")
with col2:
    search_btn = st.button("🔎 Search", use_container_width=True)

if search_btn and query.strip():
    # Apply filters
    filtered_df = df.copy()
    if selected_insurer != "All":
        filtered_df = filtered_df[filtered_df['assureur'] == selected_insurer]
    if selected_topic != "All":
        filtered_df = filtered_df[filtered_df['topic'] == selected_topic]
    if selected_sentiment != "All":
        def get_sentiment(note):
            if note <= 2: return 'Negative'
            elif note == 3: return 'Neutral'
            else: return 'Positive'
        filtered_df['sentiment'] = filtered_df['note'].apply(get_sentiment)
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    filtered_df = filtered_df[
        (filtered_df['note'] >= min_rating) & 
        (filtered_df['note'] <= max_rating)
    ]

    if filtered_df.empty:
        st.warning("No reviews match your filters.")
    else:
        # TF-IDF search on filtered subset
        sub_matrix = tfidf_matrix[filtered_df.index]
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, sub_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = filtered_df.iloc[top_indices].copy()
        results['score'] = scores[top_indices]
        results = results[results['score'] > 0]

        st.markdown(f"### 📋 {len(results)} results for *'{query}'*")

        sentiment_emoji = {5: "😊", 4: "😊", 3: "😐", 2: "😠", 1: "😠", 0: "❓"}
        sentiment_color = {5: "#10B981", 4: "#10B981", 3: "#F59E0B", 2: "#EF4444", 1: "#EF4444", 0: "#6B7280"}

        for _, row in results.iterrows():
            stars = "⭐" * int(row['note']) if row['note'] > 0 else "N/A"
            st.markdown(f"""
                <div class='review-card'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:8px'>
                        <span style='font-weight:700; color:#1F2937'>{row['assureur']}</span>
                        <span style='color:{sentiment_color.get(int(row["note"]), "#6B7280")}; font-weight:600'>
                            {stars}
                        </span>
                    </div>
                    <div style='color:#374151; margin-bottom:8px'>{str(row['avis_en'])[:300]}...</div>
                    <div style='display:flex; gap:12px'>
                        <span style='background:#F3F4F6; padding:3px 10px; border-radius:20px; font-size:12px'>
                            🏷️ {row.get('topic', 'N/A')}
                        </span>
                        <span style='background:#F3F4F6; padding:3px 10px; border-radius:20px; font-size:12px'>
                            📊 Relevance: {row['score']:.2%}
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
elif search_btn:
    st.warning("⚠️ Please enter a search query!")
else:
    st.markdown("""
        <div style='text-align:center; padding:60px; color:#9CA3AF'>
            <div style='font-size:48px'>🔎</div>
            <div style='font-size:18px; margin-top:12px'>Enter a query above to search reviews</div>
        </div>
    """, unsafe_allow_html=True)