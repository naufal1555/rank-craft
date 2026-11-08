"""
Traveloka Search Log & Position Bias Simulator (RankCraft)
Author: Ahmad Fathin Naufal (naufal1555)
Target: Traveloka / Tiket.com Search & Recommendation Data Science
Purpose:
  - Generates realistic hotel search queries with candidate pools (20 items per SERP).
  - Simulates an initial production logging policy (pi_0) with heuristic ranking.
  - Simulates eye-tracking visual examination dropoff (Position Bias: p_k = (1/k)^0.65).
  - Models user true latent relevance R(x) vs observed biased clicks C ~ Bernoulli(p_k * R).
  - Provides query-grouped train, validation, and test splits for Learning-to-Rank.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


DESTINATIONS = ["Bali", "Jakarta", "Yogyakarta", "Bandung", "Surabaya"]


@dataclass(frozen=True)
class SimulationConfig:
    n_queries: int = 2500       # 2,500 queries * 20 candidates = 50,000 ranking rows
    candidates_per_query: int = 20
    examination_decay: float = 0.65
    random_seed: int = 42
    train_ratio: float = 0.70
    test_ratio: float = 0.30


def get_examination_propensities(n_positions: int = 20, gamma: float = 0.65) -> np.ndarray:
    """Computes position examination probabilities p_k = (1 / k)^gamma."""
    positions = np.arange(1, n_positions + 1)
    p_k = (1.0 / positions) ** gamma
    return p_k.astype(np.float32)


def generate_traveloka_search_logs(config: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """
    Generates synthetic search logs with ground-truth latent utility and observed clicks.
    """
    rng = np.random.default_rng(config.random_seed)
    n_q = config.n_queries
    k = config.candidates_per_query
    total_rows = n_q * k

    # 1. Generate Query-Level Attributes
    destinations = rng.choice(DESTINATIONS, size=n_q)
    user_budget_idr = np.round(rng.lognormal(mean=13.6, sigma=0.6, size=n_q), -4)
    user_budget_idr = np.clip(user_budget_idr, 250000.0, 5000000.0)
    stay_nights = rng.choice([1, 2, 3, 5, 7], p=[0.40, 0.30, 0.15, 0.10, 0.05], size=n_q)
    guest_count = rng.choice([1, 2, 4], p=[0.25, 0.60, 0.15], size=n_q)

    # Replicate query attributes across candidate pool
    query_ids = np.repeat(np.arange(n_q), k)
    query_id_strs = [f"QRY_TVLK_{q:05d}" for q in query_ids]
    query_dest = np.repeat(destinations, k)
    query_budget = np.repeat(user_budget_idr, k)
    query_nights = np.repeat(stay_nights, k)
    query_guests = np.repeat(guest_count, k)

    # 2. Generate Candidate Hotel Features
    hotel_ids = [f"HTL_{rng.integers(1000, 9999)}" for _ in range(total_rows)]
    star_rating = rng.choice([2, 3, 4, 5], p=[0.20, 0.40, 0.25, 0.15], size=total_rows)
    review_score = np.round(rng.normal(loc=8.4, scale=0.6, size=total_rows), 1)
    review_score = np.clip(review_score, 6.0, 9.8)

    distance_to_center_km = np.round(rng.exponential(scale=4.5, size=total_rows) + 0.3, 1)
    distance_to_center_km = np.clip(distance_to_center_km, 0.2, 28.0)

    discount_pct = rng.choice([0, 10, 15, 20, 30, 40], p=[0.30, 0.25, 0.20, 0.15, 0.07, 0.03], size=total_rows)
    amenities_count = rng.integers(2, 12, size=total_rows)

    base_price_by_star = {2: 320000, 3: 580000, 4: 1100000, 5: 2400000}
    hotel_price_idr = np.array([
        base_price_by_star[s] * rng.uniform(0.75, 1.35) * (1.0 - d / 100.0)
        for s, d in zip(star_rating, discount_pct)
    ])
    hotel_price_idr = np.round(hotel_price_idr, -3)

    budget_affinity_ratio = np.round(hotel_price_idr / query_budget, 2)
    historical_ctr = np.clip(rng.beta(a=3, b=25, size=total_rows), 0.01, 0.25)

    # 3. Production Logging Policy pi_0 (Past Ranker)
    # The existing production search engine sorts items primarily by historical CTR and star rating
    # with exploratory random perturbation
    logging_heuristic = (
        4.0 * historical_ctr
        + 0.30 * star_rating
        + 0.05 * (review_score - 7.0)
        + rng.normal(0, 0.25, size=total_rows)
    )

    # Assign display ranks 1..k within each query according to pi_0
    display_ranks = np.zeros(total_rows, dtype=int)
    for q in range(n_q):
        s_idx = q * k
        e_idx = s_idx + k
        q_heur = logging_heuristic[s_idx:e_idx]
        order = np.argsort(q_heur)[::-1]
        ranks = np.empty(k, dtype=int)
        ranks[order] = np.arange(1, k + 1)
        display_ranks[s_idx:e_idx] = ranks

    # Visual examination probability based on assigned position: p_k = (1 / k)^gamma
    exam_probs = (1.0 / display_ranks.astype(np.float32)) ** config.examination_decay

    # 4. Compute True Latent User Utility R(x)
    # The true conversion relevance of the hotel to the user, free of visual position bias!
    affinity_penalty = np.abs(np.log(np.clip(budget_affinity_ratio, 0.2, 5.0)))
    raw_utility = (
        0.70 * (star_rating - 3.0)
        + 1.10 * (review_score - 8.0)
        - 1.30 * affinity_penalty
        - 0.08 * distance_to_center_km
        + 0.03 * discount_pct
        + 0.15 * amenities_count
    )
    # True latent relevance in [0.02, 0.95]
    true_relevance_prob = 1.0 / (1.0 + np.exp(-raw_utility))
    true_relevance_prob = np.clip(true_relevance_prob, 0.02, 0.95)

    # 5. Position-Based Click Simulation: C_i ~ Bernoulli(p_k * R_i)
    click_prob = exam_probs * true_relevance_prob
    observed_click = (rng.random(size=total_rows) < click_prob).astype(int)

    df = pd.DataFrame({
        "query_id": query_id_strs,
        "query_dest": query_dest,
        "query_budget_idr": query_budget,
        "query_nights": query_nights,
        "query_guests": query_guests,
        "hotel_id": hotel_ids,
        "display_rank": display_ranks,
        "exam_prob_p_k": np.round(exam_probs, 4),
        "star_rating": star_rating,
        "review_score": review_score,
        "distance_to_center_km": distance_to_center_km,
        "discount_pct": discount_pct,
        "amenities_count": amenities_count,
        "hotel_price_idr": hotel_price_idr,
        "budget_affinity_ratio": budget_affinity_ratio,
        "historical_ctr": np.round(historical_ctr, 4),
        "true_relevance_score": np.round(true_relevance_prob, 4),
        "observed_click": observed_click,
    })

    return df


def split_queries_train_test(
    df: pd.DataFrame,
    config: SimulationConfig = SimulationConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into train and test sets by query_id."""
    unique_queries = df["query_id"].unique()
    n_train = int(len(unique_queries) * config.train_ratio)

    train_q = set(unique_queries[:n_train])
    test_q = set(unique_queries[n_train:])

    train_df = df[df["query_id"].isin(train_q)].copy().reset_index(drop=True)
    test_df = df[df["query_id"].isin(test_q)].copy().reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    print("[TEST] Generating 50,000 Traveloka search interaction logs...")
    cfg = SimulationConfig(n_queries=2500)
    df = generate_traveloka_search_logs(cfg)
    print(f"Generated {len(df):,} ranking candidates across {cfg.n_queries:,} search queries.")
    tr_df, te_df = split_queries_train_test(df, cfg)
    print(f"Query Split -> Train: {tr_df['query_id'].nunique():,} queries | Test: {te_df['query_id'].nunique():,} queries")
