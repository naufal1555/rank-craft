"""
Counterfactual Learning-to-Rank Evaluation & Benchmarking Engine (RankCraft)
Author: Ahmad Fathin Naufal (naufal1555)
Purpose:
  - Computes Unbiased NDCG@K against ground-truth latent user utilities.
  - Measures Rank Correlation (Kendall's Tau & Spearman) against true relevance.
  - Benchmarks Naive Ranker vs IPS Ranker vs Doubly Robust Ranker vs Random.
  - Generates high-resolution comparative NDCG benchmark plot.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr


def compute_dcg_at_k(relevance_array: np.ndarray, k: int) -> float:
    """Computes Discounted Cumulative Gain at rank cutoff k."""
    rel_k = relevance_array[:k]
    discounts = np.log2(np.arange(2, len(rel_k) + 2))
    # Standard formula: (2^rel - 1) / log2(i + 1)
    gains = (2.0 ** rel_k) - 1.0
    return float(np.sum(gains / discounts))


def compute_query_ndcg(
    true_relevance: np.ndarray,
    predicted_scores: np.ndarray,
    k: int = 10,
) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG@K) for a single query."""
    # Rank items according to predicted scores (descending)
    pred_order = np.argsort(predicted_scores)[::-1]
    sorted_true_rel_by_pred = true_relevance[pred_order]
    actual_dcg = compute_dcg_at_k(sorted_true_rel_by_pred, k)

    # Ideal ranking (sorted strictly by ground truth relevance)
    ideal_order = np.argsort(true_relevance)[::-1]
    sorted_true_rel_ideal = true_relevance[ideal_order]
    ideal_dcg = compute_dcg_at_k(sorted_true_rel_ideal, k)

    if ideal_dcg == 0.0:
        return 1.0
    return float(actual_dcg / ideal_dcg)


def evaluate_ranking_policy(
    test_df: pd.DataFrame,
    score_col_or_array: Any,
    policy_name: str,
    k_list: List[int] = [3, 5, 10, 20],
) -> Dict[str, Any]:
    """
    Evaluates a ranking policy across all test queries:
      - Mean NDCG@K for each cutoff in k_list
      - Mean Kendall's Tau correlation with true relevance
    """
    if isinstance(score_col_or_array, str):
        scores = test_df[score_col_or_array].values
    else:
        scores = np.asarray(score_col_or_array)

    query_groups = test_df.groupby("query_id", sort=False)
    
    ndcg_records = {f"ndcg@{k}": [] for k in k_list}
    tau_records = []

    for _, group in query_groups:
        true_rel = group["true_relevance_score"].values
        pred_scores = scores[group.index]

        for k in k_list:
            ndcg = compute_query_ndcg(true_rel, pred_scores, k=k)
            ndcg_records[f"ndcg@{k}"].append(ndcg)

        # Kendall Tau
        if len(np.unique(pred_scores)) > 1:
            tau, _ = kendalltau(true_rel, pred_scores)
            if not np.isnan(tau):
                tau_records.append(tau)

    result = {
        "policy": policy_name,
        **{k: round(float(np.mean(vals)), 4) for k, vals in ndcg_records.items()},
        "kendall_tau": round(float(np.mean(tau_records)), 4) if tau_records else 0.0,
    }
    return result


def plot_ndcg_benchmark(
    benchmark_results: List[Dict[str, Any]],
    save_path: str,
):
    """Generates comparison bar plot of NDCG@5 and NDCG@10 across rankers."""
    policies = [r["policy"] for r in benchmark_results]
    ndcg5 = [r["ndcg@5"] for r in benchmark_results]
    ndcg10 = [r["ndcg@10"] for r in benchmark_results]

    x = np.arange(len(policies))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    bars1 = plt.bar(x - width / 2, ndcg5, width, label="Unbiased NDCG@5", color="#1E88E5", alpha=0.90)
    bars2 = plt.bar(x + width / 2, ndcg10, width, label="Unbiased NDCG@10", color="#43A047", alpha=0.90)

    # Add text labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title("RankCraft: Unbiased Learning-to-Rank Benchmark (Traveloka SERP)", fontsize=13, fontweight="bold")
    plt.ylabel("Normalized Discounted Cumulative Gain (NDCG)", fontsize=11)
    plt.xticks(x, policies, fontsize=10, fontweight="medium")
    plt.ylim(0.40, 1.02)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    from dataset import SimulationConfig, generate_traveloka_search_logs, split_queries_train_test
    from models import NaiveRanker, IPSRanker, DoublyRobustRanker

    cfg = SimulationConfig(n_queries=1000)
    df = generate_traveloka_search_logs(cfg)
    tr, te = split_queries_train_test(df, cfg)

    naive = NaiveRanker()
    naive.fit(tr)
    te["score_naive"] = naive.predict_scores(te)

    ips = IPSRanker()
    ips.fit(tr)
    te["score_ips"] = ips.predict_scores(te)

    dr = DoublyRobustRanker()
    dr.fit(tr)
    te["score_dr"] = dr.predict_scores(te)

    # Add random baseline
    rng = np.random.default_rng(42)
    te["score_random"] = rng.random(len(te))

    res_rand = evaluate_ranking_policy(te, "score_random", "Random Order")
    res_naive = evaluate_ranking_policy(te, "score_naive", "Naive Ranker (Biased)")
    res_ips = evaluate_ranking_policy(te, "score_ips", "IPS Ranker (Debiased)")
    res_dr = evaluate_ranking_policy(te, "score_dr", "Doubly Robust (RankCraft)")

    eval_df = pd.DataFrame([res_rand, res_naive, res_ips, res_dr])
    print(eval_df)
