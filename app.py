import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Insurance Review Analytics",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .main { background-color: #FFF0F3; }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero {
        background: linear-gradient(135deg, #8B1A4A 0%, #C2185B 50%, #E91E8C 100%);
        border-radius: 24px;
        padding: 60px 40px;
        text-align: center;
        margin-bottom: 40px;
        color: white;
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: -1px;
    }
    .hero p {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        letter-spacing: 0.5px;
    }
    
    .stat-card {
        background-color: white;
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        box-shadow: 0 2px 20px rgba(139,26,74,0.08);
        border-top: 3px solid #C2185B;
    }
    .stat-label {
        font-size: 11px;
        color: #9CA3AF;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #8B1A4A;
        margin: 6px 0 0 0;
        font-family: 'Playfair Display', serif;
    }
    
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: #1F2937;
        margin: 40px 0 20px 0;
        text-align: center;
    }
    
    .app-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 20px;
    }
    
    .app-card {
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 20px rgba(139,26,74,0.06);
        transition: transform 0.2s;
        border-bottom: 3px solid #FFF0F3;
    }
    .app-card:hover {
        transform: translateY(-4px);
        border-bottom: 3px solid #C2185B;
    }
    .app-emoji {
        font-size: 2rem;
        margin-bottom: 12px;
    }
    .app-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 8px;
        font-family: 'Playfair Display', serif;
    }
    .app-desc {
        font-size: 0.85rem;
        color: #6B7280;
        line-height: 1.5;
    }
    
    .footer {
        text-align: center;
        color: #9CA3AF;
        font-size: 13px;
        margin-top: 60px;
        padding: 20px;
        border-top: 1px solid #FCE4EC;
    }
    
    .badge {
        display: inline-block;
        background: #FCE4EC;
        color: #C2185B;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
    <div class='hero'>
        <h1>Insurance Review Analytics</h1>
        <p>A complete NLP platform for analyzing 34,000+ French insurance customer reviews</p>
        <div style='margin-top:20px'>
            <span class='badge'>🤖 Machine Learning</span>
            <span class='badge'>💬 NLP</span>
            <span class='badge'>📊 Data Analytics</span>
            <span class='badge'>🔍 Semantic Search</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Stats
@st.cache_data
def load_stats():
    df = pd.read_csv("topic_dataset.csv")
    return df

df = load_stats()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Total Reviews</div>
            <div class='stat-value'>{len(df):,}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Insurers</div>
            <div class='stat-value'>{df['assureur'].nunique()}</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Avg Rating</div>
            <div class='stat-value'>{df['note'].mean():.2f} ⭐</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Topics</div>
            <div class='stat-value'>6</div>
        </div>
    """, unsafe_allow_html=True)

# Apps
st.markdown("<div class='section-title'>Our Applications</div>", unsafe_allow_html=True)

apps = [
    ("🔮", "Prediction", "Enter a review and instantly get its predicted star rating, sentiment and topic with confidence scores."),
    ("📋", "Summary", "Explore insurer performance with AI-generated summaries, rating distributions and topic breakdowns."),
    ("🔍", "Explanation", "Understand why the model made its prediction — see which words drove each decision."),
    ("🔎", "Search", "Search through 34,000+ reviews using keywords or semantic queries with advanced filters."),
    ("🤖", "RAG", "Ask any question about insurance reviews and get answers powered by Retrieval Augmented Generation."),
    ("💬", "Q&A", "Get precise answers extracted directly from real customer reviews.")
]

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3, col1, col2, col3]

for i, (emoji, name, description) in enumerate(apps):
    with cols[i]:
        st.markdown(f"""
            <div class='app-card'>
                <div class='app-emoji'>{emoji}</div>
                <div class='app-name'>{name}</div>
                <div class='app-desc'>{description}</div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        © 2026 Insurance Review Analytics. All rights reserved. | Built by Dovila Longmis
    </div>
""", unsafe_allow_html=True)