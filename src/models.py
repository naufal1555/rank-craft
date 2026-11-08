"""
Counterfactual Learning-to-Rank & Debiasing Models (RankCraft)
Author: Ahmad Fathin Naufal (naufal1555)
Reference: Joachims et al. (KDD 2017) "Unbiased Learning-to-Rank with Biased Feedback"
Purpose:
  - Naive Ranker: Standard supervised model trained on biased clicks (Baseline).
  - IPS Ranker: Inverse Propensity Scoring ranker with propensity weight clipping.
  - Doubly Robust Ranker: Low-variance counterfactual estimator with imputed baseline.
  - Contextual Thompson Sampling: Multi-armed bandit for exploratory cold-start hotel ranking.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import xgboost as xgb


RANKING_FEATURES = [
    "star_rating",
    "review_score",
    "distance_to_center_km",
    "discount_pct",
    "amenities_count",
    "hotel_price_idr",
    "budget_affinity_ratio",
    "historical_ctr",
]


class NaiveRanker:
    """Standard supervised ranker trained directly on observed clicks without debiasing."""

    def __init__(self, random_seed: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_seed,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame):
        X = df[RANKING_FEATURES]
        y = df["observed_click"].values
        self.model.fit(X, y)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[RANKING_FEATURES])[:, 1]


class IPSRanker:
    """
    Inverse Propensity Scoring (IPS) Counterfactual Ranker.
    Weights each click observation by w_i = 1 / p_k to eliminate visual position bias.
    """

    def __init__(self, clip_min: float = 0.12, random_seed: int = 42):
        self.clip_min = clip_min
        self.model = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_seed,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame):
        X = df[RANKING_FEATURES]
        y = df["observed_click"].values
        p_k = np.clip(df["exam_prob_p_k"].values, self.clip_min, 1.0)
        
        # Sample weights: 1 / p_k for positive clicks, normalized
        weights = np.where(y == 1, 1.0 / p_k, 1.0)
        # Normalize weights to prevent gradient explosion
        weights = weights / np.mean(weights)

        self.model.fit(X, y, sample_weight=weights)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[RANKING_FEATURES])[:, 1]


class DoublyRobustRanker:
    """
    Doubly Robust (DR) Counterfactual Ranker.
    Imputes baseline relevance g(x) and combines it with IPS-weighted residuals.
    Guarantees unbiasedness with significantly lower variance than pure IPS.
    """

    def __init__(self, clip_min: float = 0.12, random_seed: int = 42):
        self.clip_min = clip_min
        self.imputation_model = xgb.XGBRegressor(
            n_estimators=60,
            learning_rate=0.08,
            max_depth=3,
            random_state=random_seed,
            n_jobs=-1,
        )
        self.final_ranker = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_seed + 1,
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame):
        X = df[RANKING_FEATURES]
        y = df["observed_click"].values
        p_k = np.clip(df["exam_prob_p_k"].values, self.clip_min, 1.0)

        # Stage 1: Fit baseline imputation model on observed clicks
        self.imputation_model.fit(X, y)
        r_hat = np.clip(self.imputation_model.predict(X), 0.0, 1.0)

        # Stage 2: Construct Doubly Robust pseudo-labels
        # DR label: r_hat + (C - r_hat * p_k) / p_k
        dr_pseudo_labels = r_hat + (y - (r_hat * p_k)) / p_k
        dr_pseudo_labels = np.clip(dr_pseudo_labels, 0.0, 3.0)

        # Stage 3: Train final ranker on DR pseudo-labels
        self.final_ranker.fit(X, dr_pseudo_labels)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.final_ranker.predict(df[RANKING_FEATURES])


class ContextualThompsonSamplingBandit:
    """
    Exploration & Discovery Engine for Cold-Start / New Hotels.
    Draws Bayesian Beta posterior samples for exploratory slots.
    """

    def __init__(self, exploration_weight: float = 0.35, random_seed: int = 42):
        self.exploration_weight = exploration_weight
        self.rng = np.random.default_rng(random_seed)

    def rerank_with_exploration(
        self,
        base_scores: np.ndarray,
        impression_counts: np.ndarray,
        click_counts: np.ndarray,
        prior_alpha: float = 2.0,
        prior_beta: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates Thompson Sampling exploration bonus for candidates:
          theta ~ Beta(prior_alpha + clicks, prior_beta + (impressions - clicks))
        Returns:
          final_scores: blended score for ranking
          is_explored: boolean array indicating items promoted by exploration
        """
        n = len(base_scores)
        alphas = prior_alpha + np.maximum(0, click_counts)
        betas = prior_beta + np.maximum(0, impression_counts - click_counts)

        # Draw posterior samples
        thompson_samples = self.rng.beta(alphas, betas)

        # Identify cold-start candidates (< 30 impressions)
        is_cold_start = impression_counts < 30
        
        # Normalize base scores to [0, 1]
        s_min, s_max = np.min(base_scores), np.max(base_scores)
        norm_scores = (base_scores - s_min) / (s_max - s_min + 1e-6)

        # Blend: cold-start gets higher exploration boost
        boost = np.where(is_cold_start, self.exploration_weight * thompson_samples, 0.0)
        final_scores = norm_scores + boost

        return final_scores, is_cold_start


if __name__ == "__main__":
    from dataset import SimulationConfig, generate_traveloka_search_logs, split_queries_train_test

    print("[TEST] Training Naive, IPS, and Doubly Robust Rankers...")
    cfg = SimulationConfig(n_queries=1000)
    df = generate_traveloka_search_logs(cfg)
    tr, te = split_queries_train_test(df, cfg)

    print("1. Training Naive Ranker...")
    naive = NaiveRanker()
    naive.fit(tr)

    print("2. Training IPS Ranker...")
    ips = IPSRanker()
    ips.fit(tr)

    print("3. Training Doubly Robust Ranker...")
    dr = DoublyRobustRanker()
    dr.fit(tr)

    print("\nSample Predictions on first query candidates:")
    sample_q = te[te["query_id"] == te["query_id"].iloc[0]]
    s_naive = naive.predict_scores(sample_q)
    s_ips = ips.predict_scores(sample_q)
    s_dr = dr.predict_scores(sample_q)

    comp_df = pd.DataFrame({
        "hotel_id": sample_q["hotel_id"].values[:5],
        "true_rel": sample_q["true_relevance_score"].values[:5],
        "naive_score": np.round(s_naive[:5], 3),
        "ips_score": np.round(s_ips[:5], 3),
        "dr_score": np.round(s_dr[:5], 3),
    })
    print(comp_df)
