import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import defaultdict

st.set_page_config(page_title="Insurer Summary", page_icon="📋", layout="wide")

st.title("📋 Insurer Review Summary")
st.markdown("Select an insurer to get a summary of their reviews and performance metrics.")

@st.cache_data
def load_data():
    df = pd.read_csv("topic_dataset.csv")
    df = df.dropna(subset=['avis_en', 'assureur'])
    return df

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

df = load_data()
summarizer = load_summarizer()

# Sidebar filters
st.sidebar.header("🔍 Filters")
insurers = sorted(df['assureur'].unique().tolist())
selected_insurer = st.sidebar.selectbox("Select Insurer", insurers)
selected_topic = st.sidebar.selectbox("Filter by Topic", ["All"] + sorted(df['topic'].dropna().unique().tolist()))
selected_rating = st.sidebar.slider("Minimum Rating", 1, 5, 1)

# Filter data
filtered = df[df['assureur'] == selected_insurer]
if selected_topic != "All":
    filtered = filtered[filtered['topic'] == selected_topic]
filtered = filtered[filtered['note'] >= selected_rating]

st.subheader(f"📊 Overview — {selected_insurer}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", len(filtered))
col2.metric("Avg Rating", f"{filtered['note'].mean():.2f} ⭐" if not filtered.empty else "N/A")
col3.metric("Positive Reviews", f"{(filtered['note'] >= 4).sum()}")
col4.metric("Negative Reviews", f"{(filtered['note'] <= 2).sum()}")

# Rating distribution
if not filtered.empty:
    st.subheader("⭐ Rating Distribution")
    rating_counts = filtered['note'].value_counts().sort_index()
    st.bar_chart(rating_counts)

    # Topic distribution
    st.subheader("🏷️ Topic Distribution")
    topic_counts = filtered['topic'].value_counts()
    st.bar_chart(topic_counts)

    # Summary generation
    st.subheader("📝 AI-Generated Summary")
    if st.button("Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            sample_reviews = filtered['avis_en'].dropna().head(20).tolist()
            combined = " ".join(sample_reviews)[:1000]
            if len(combined.split()) > 30:
                summary = summarizer(combined, max_length=100, min_length=30, do_sample=False)
                st.info(summary[0]['summary_text'])
            else:
                st.info(combined)

    # Sample reviews
    st.subheader("💬 Sample Reviews")
    st.dataframe(
        filtered[['note', 'topic', 'avis_en']].head(10).rename(
            columns={'note': 'Rating', 'topic': 'Topic', 'avis_en': 'Review'}
        ),
        use_container_width=True
    )
else:
    st.warning("No reviews found for the selected filters.")