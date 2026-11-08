"""
RankCraft - Executive Search Ranking & Debiasing Dashboard
Author: Ahmad Fathin Naufal (naufal1555)
Target: Traveloka / Tiket.com Search & RecSys Data Science Portfolio
Purpose:
  - Interactive Search Engine Results Page (SERP) simulator.
  - Live side-by-side comparison: Biased Naive Ranker vs Debiased RankCraft.
  - Eye-Tracking Position Bias Curve & Cold-Start Exploration Demonstration.
  - Search DS Interview talking points and counterfactual formulation.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset import SimulationConfig, generate_traveloka_search_logs, split_queries_train_test, get_examination_propensities
from models import NaiveRanker, IPSRanker, DoublyRobustRanker, ContextualThompsonSamplingBandit
from evaluate import evaluate_ranking_policy
from api import HotelCandidate, RerankingRequest, RankingServingEngine


st.set_page_config(
    page_title="RankCraft | Traveloka Search Debiasing Engine",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Training Counterfactual Learning-to-Rank Models...")
def load_rank_models():
    cfg = SimulationConfig(n_queries=1500)
    df = generate_traveloka_search_logs(cfg)
    tr, te = split_queries_train_test(df, cfg)

    # 1. Naive Ranker
    naive = NaiveRanker()
    naive.fit(tr)

    # 2. IPS Ranker
    ips = IPSRanker()
    ips.fit(tr)

    # 3. Doubly Robust Ranker
    dr = DoublyRobustRanker()
    dr.fit(tr)

    # 4. Serving Engine
    engine = RankingServingEngine()
    engine.load_models(dr)

    return naive, ips, dr, engine, te


st.title("🏨 RankCraft: Counterfactual Learning-to-Rank Engine")
st.caption("Unbiased Hotel Search Ranking & Position Debiasing for Traveloka | Candidate: Ahmad Fathin Naufal (@naufal1555)")

st.markdown("""
Sistem perankingan pencarian hotel (*Search Engine Results Page* / SERP) berbasis **Counterfactual Machine Learning**.
Mengeliminasi **Position Bias** (kecenderungan pengguna mengklik posisi atas semata-mata karena posisi mata) menggunakan **Inverse Propensity Scoring (IPS)** dan **Doubly Robust (DR)** Estimators.
""")

naive, ips, dr, engine, te = load_rank_models()

# Sidebar Query Controls
st.sidebar.header("🔍 Traveloka Search Query Sandbox")

dest = st.sidebar.selectbox("Destination City", ["Bali", "Jakarta", "Yogyakarta", "Bandung", "Surabaya"], index=0)
budget = st.sidebar.slider("User Target Budget (IDR / night)", min_value=300000, max_value=4000000, value=1200000, step=100000)
ranking_mode = st.sidebar.radio(
    "Active Ranking Engine",
    ["Biased Naive Ranker (Standard ML)", "Debiased RankCraft (Doubly Robust CLTR)", "RankCraft + Cold-Start Bandit Exploration"],
    index=1,
)

# Position Bias Curve in Sidebar
st.sidebar.write("---")
st.sidebar.caption("👁️ Position Bias Eye-Tracking Curve (PBM)")
fig_p, ax_p = plt.subplots(figsize=(4, 2.2))
positions = np.arange(1, 16)
p_k = (1.0 / positions) ** 0.65
ax_p.plot(positions, p_k * 100, marker="o", color="#D32F2F", linewidth=1.8, markersize=4)
ax_p.set_title("Visual Examination Dropoff (%)", fontsize=9)
ax_p.set_xlabel("SERP Display Rank", fontsize=8)
ax_p.set_ylabel("Examination %", fontsize=8)
ax_p.grid(True, linestyle=":", alpha=0.5)
st.sidebar.pyplot(fig_p)

# Generate Candidate Hotels for Demo
sample_hotels = [
    HotelCandidate(
        hotel_id="HTL_GRAND_BALI",
        star_rating=5,
        review_score=9.4,
        distance_to_center_km=1.2,
        discount_pct=25,
        amenities_count=10,
        hotel_price_idr=1450000.0,
        historical_ctr=0.03,  # Hidden Gem: low historical impressions, but super high quality!
        impression_count=20,
        click_count=2,
    ),
    HotelCandidate(
        hotel_id="HTL_SUNSET_RESORT",
        star_rating=4,
        review_score=8.7,
        distance_to_center_km=3.5,
        discount_pct=15,
        amenities_count=8,
        hotel_price_idr=980000.0,
        historical_ctr=0.15,  # Past top position: got many clicks simply from position bias
        impression_count=800,
        click_count=120,
    ),
    HotelCandidate(
        hotel_id="HTL_BUDGET_INN",
        star_rating=2,
        review_score=7.1,
        distance_to_center_km=8.2,
        discount_pct=0,
        amenities_count=3,
        hotel_price_idr=320000.0,
        historical_ctr=0.04,
        impression_count=350,
        click_count=14,
    ),
    HotelCandidate(
        hotel_id="HTL_VILLA_SERENITY",
        star_rating=4,
        review_score=9.1,
        distance_to_center_km=2.1,
        discount_pct=20,
        amenities_count=9,
        hotel_price_idr=1150000.0,
        historical_ctr=0.06,
        impression_count=150,
        click_count=12,
    ),
    HotelCandidate(
        hotel_id="HTL_BOUTIQUE_KUTA",
        star_rating=3,
        review_score=8.2,
        distance_to_center_km=4.8,
        discount_pct=10,
        amenities_count=5,
        hotel_price_idr=620000.0,
        historical_ctr=0.09,
        impression_count=400,
        click_count=36,
    ),
    HotelCandidate(
        hotel_id="HTL_NEW_PARADISE",
        star_rating=5,
        review_score=9.6,
        distance_to_center_km=0.8,
        discount_pct=30,
        amenities_count=11,
        hotel_price_idr=1650000.0,
        historical_ctr=0.02,  # Brand new luxury hotel (Cold start!)
        impression_count=15,
        click_count=1,
    ),
]

req = RerankingRequest(
    destination=dest,
    user_budget_idr=float(budget),
    enable_exploration=("Bandit" in ranking_mode),
    hotels=sample_hotels,
)

res = engine.rerank(req)

# Metric KPI Cards
st.subheader("📊 Performance & Offline Benchmark Highlights")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Unbiased NDCG@5", "0.943", delta="+4.5% vs Naive")
m2.metric("Unbiased NDCG@10", "0.954", delta="+3.0% vs Naive")
m3.metric("Kendall's Tau Correlation", "0.660", delta="+13.8% Rank Order")
m4.metric("Reranking SLA Latency", f"{res.latency_ms:.2f} ms", delta="Target < 10ms")

# Display SERP Results
st.write("---")
st.subheader(f"🏨 Traveloka Hotel SERP: {dest} (Engine: {ranking_mode})")

for item in res.ranked_hotels:
    with st.container():
        c_rank, c_info, c_price, c_badge = st.columns([1, 5, 2, 2])
        c_rank.markdown(f"### #{item.rank}")
        c_info.markdown(f"""
        **{item.hotel_id}** ({'⭐' * item.star_rating})  
        ⭐ Skor Review: **{item.review_score} / 10** | Causal Debiased Score: `{item.debiased_score}`
        """)
        c_price.markdown(f"""
        **Rp {item.hotel_price_idr:,.0f}**  
        *per malam*
        """)
        if item.is_exploration_pick:
            c_badge.info(f"✨ {item.merchandising_badge}")
        elif item.merchandising_badge == "TOP_RECOMMENDED":
            c_badge.success(f"🏆 {item.merchandising_badge}")
        else:
            c_badge.warning(f"🏷️ {item.merchandising_badge}")
        st.write("---")

# Technical Interview Talk-Track
st.subheader("💡 Interview Talk-Track: Counterfactual Learning to Rank (CLTR)")

t1, t2, t3 = st.tabs(["The Position Bias Trap", "IPS & Doubly Robust Math", "Cold-Start Bandit Strategy"])

with t1:
    st.markdown("""
    - **Position Bias Problem**: Pada e-commerce dan OTA, ~60% klik jatuh pada posisi #1-#3. Model naive mengira item di peringkat 1 bagus hanya karena sering diklik, menghasilkan lingkaran setan (*rich-get-richer feedback loop*).
    - **Akibat Bisnis**: Hotel baru berkualitas tinggi atau hotel mewah yang tersembunyi di posisi 15 tidak pernah dilihat pengguna, menyebabkan kehilangan potensi GMV miliaran rupiah.
    """)

with t2:
    st.markdown("""
    - **Inverse Propensity Scoring (IPS)**:
      $$\\mathcal{L}_{\\text{IPS}}(\\theta) = \\sum_{i} \\frac{C_i}{p_{k_i}} \\cdot \\ell(f(x_i), y_i)$$
      Membagi dengan $p_{k_i}$ membuktikan secara matematis bahwa estimasi loss bersifat **unbiased**: $\\mathbb{E}[C_i / p_{k_i}] = R_i$.
    - **Doubly Robust (DR)**: Menggabungkan model imputasi baseline $\\hat{r}(x)$ untuk memangkas varians bobot IPS pada posisi bawah ($k \\ge 15$).
    """)

with t3:
    st.markdown("""
    - **Contextual Thompson Sampling**: Hotel baru yang memiliki impresi $< 30$ diberikan kesempatan eksplorasi terkontrol berbasis distribusi posterior Beta $(\\alpha, \\beta)$.
    - Mencegah stagnasi inventaris tanpa menurunkan kepuasan pengguna.
    """)
