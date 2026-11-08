"""
RankCraft End-to-End Pipeline & Evaluation Benchmark
Author: Ahmad Fathin Naufal (naufal1555)
Target: Traveloka / Tiket.com Search & Discovery Data Science
Purpose:
  - Generates 50,000 simulated hotel search logs with position bias.
  - Trains Naive, IPS, and Doubly Robust Counterfactual Rankers.
  - Evaluates Unbiased NDCG@K and Kendall's Tau against ground truth utilities.
  - Generates artifacts/unbiased_ndcg_benchmark.png.
  - Benchmarks Real-Time Serving API latency (< 10ms target).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from dataset import SimulationConfig, generate_traveloka_search_logs, split_queries_train_test
from models import (
    RANKING_FEATURES,
    NaiveRanker,
    IPSRanker,
    DoublyRobustRanker,
    ContextualThompsonSamplingBandit,
)
from evaluate import evaluate_ranking_policy, plot_ndcg_benchmark
from api import HotelCandidate, RerankingRequest, RankingServingEngine


def main():
    print("=" * 80)
    print(" [RANKCRAFT] COUNTERFACTUAL LEARNING-TO-RANK & POSITION DEBIASING ENGINE")
    print(" Candidate: Ahmad Fathin Naufal (naufal1555)")
    print(" Target: Traveloka / Tiket.com Search & RecSys Data Science")
    print("=" * 80)

    # 1. Dataset Generation
    print("\n[STEP 1] Generating Simulated Traveloka Search Logs (50,000 rows)...")
    cfg = SimulationConfig(n_queries=2500, random_seed=42)
    df = generate_traveloka_search_logs(cfg)
    train_df, test_df = split_queries_train_test(df, cfg)
    print(f"  -> Total Search Rows: {len(df):,} across {cfg.n_queries:,} queries (20 items/SERP)")
    print(f"  -> Split: Train = {train_df['query_id'].nunique():,} queries ({len(train_df):,} rows) | Test = {test_df['query_id'].nunique():,} queries ({len(test_df):,} rows)")

    # Print position bias evidence
    clicks_by_rank = df.groupby("display_rank")["observed_click"].mean()
    print("  -> Logged Position Bias Evidence:")
    print(f"     - Rank #1 CTR : {clicks_by_rank[1]*100:>5.2f}% (High visual saliency)")
    print(f"     - Rank #5 CTR : {clicks_by_rank[5]*100:>5.2f}%")
    print(f"     - Rank #10 CTR: {clicks_by_rank[10]*100:>5.2f}%")
    print(f"     - Rank #20 CTR: {clicks_by_rank[20]*100:>5.2f}% (Visual dropoff)")

    # 2. Model Training
    print("\n[STEP 2] Training Rankers...")
    
    # 2a. Naive Ranker
    t0 = time.time()
    naive = NaiveRanker(random_seed=42)
    naive.fit(train_df)
    t_naive = time.time() - t0
    print(f"  [OK] Naive Ranker (Biased XGBoost) trained in {t_naive:.2f}s")

    # 2b. IPS Ranker
    t0 = time.time()
    ips = IPSRanker(clip_min=0.12, random_seed=42)
    ips.fit(train_df)
    t_ips = time.time() - t0
    print(f"  [OK] IPS Counterfactual Ranker (Inverse Propensity) trained in {t_ips:.2f}s")

    # 2c. Doubly Robust Ranker
    t0 = time.time()
    dr = DoublyRobustRanker(clip_min=0.12, random_seed=42)
    dr.fit(train_df)
    t_dr = time.time() - t0
    print(f"  [OK] Doubly Robust Counterfactual Ranker (Imputation + Residuals) trained in {t_dr:.2f}s")

    # 3. Model Predictions on Test Set
    print("\n[STEP 3] Generating Predictions on Out-of-Sample Test Set (750 queries)...")
    test_df["score_naive"] = naive.predict_scores(test_df)
    test_df["score_ips"] = ips.predict_scores(test_df)
    test_df["score_dr"] = dr.predict_scores(test_df)

    rng = np.random.default_rng(42)
    test_df["score_random"] = rng.random(len(test_df))

    # 4. Evaluation & Benchmarking
    print("\n[STEP 4] Computing Unbiased NDCG & Rank Correlation Metrics...")
    eval_random = evaluate_ranking_policy(test_df, "score_random", "Random Ranking")
    eval_naive = evaluate_ranking_policy(test_df, "score_naive", "Naive Ranker (Biased)")
    eval_ips = evaluate_ranking_policy(test_df, "score_ips", "IPS Ranker (Debiased)")
    eval_dr = evaluate_ranking_policy(test_df, "score_dr", "Doubly Robust (RankCraft)")

    benchmark_records = [eval_random, eval_naive, eval_ips, eval_dr]

    print("-" * 80)
    print(f"{'Ranking Policy':<28} | {'NDCG@3':<9} | {'NDCG@5':<9} | {'NDCG@10':<9} | {'Kendall Tau':<11}")
    print("-" * 80)
    for r in benchmark_records:
        print(f"{r['policy']:<28} | {r['ndcg@3']:>7.4f}  | {r['ndcg@5']:>7.4f}  | {r['ndcg@10']:>7.4f}  | {r['kendall_tau']:>9.4f}")
    print("-" * 80)

    base_ndcg5 = eval_naive["ndcg@5"]
    dr_ndcg5 = eval_dr["ndcg@5"]
    ips_ndcg5 = eval_ips["ndcg@5"]
    lift_dr = ((dr_ndcg5 - base_ndcg5) / base_ndcg5) * 100.0
    lift_ips = ((ips_ndcg5 - base_ndcg5) / base_ndcg5) * 100.0
    print(f"  -> Relative NDCG@5 Lift (IPS over Naive): +{lift_ips:.2f}%")
    print(f"  -> Relative NDCG@5 Lift (Doubly Robust over Naive): +{lift_dr:.2f}%")

    # 5. Save Plot to Artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    plot_path = os.path.join(artifacts_dir, "unbiased_ndcg_benchmark.png")
    plot_ndcg_benchmark(benchmark_records, plot_path)
    print(f"\n[OK] Benchmark NDCG Plot saved to {plot_path}")

    # 6. Real-Time Serving API Benchmark
    print("\n[STEP 5] Benchmarking Real-Time Reranking Serving Engine (< 10ms target)...")
    engine = RankingServingEngine()
    engine.load_models(dr)

    # 20 candidate hotels for test query
    sample_hotels = [
        HotelCandidate(
            hotel_id=f"HTL_TEST_{i:03d}",
            star_rating=int(np.random.choice([2, 3, 4, 5])),
            review_score=round(float(np.random.uniform(7.0, 9.8)), 1),
            distance_to_center_km=round(float(np.random.uniform(0.5, 12.0)), 1),
            discount_pct=int(np.random.choice([0, 10, 15, 20, 30])),
            amenities_count=int(np.random.randint(3, 11)),
            hotel_price_idr=round(float(np.random.uniform(350000, 2500000)), -3),
            historical_ctr=round(float(np.random.uniform(0.02, 0.15)), 3),
            impression_count=int(np.random.randint(10, 500)),
            click_count=int(np.random.randint(1, 40)),
        )
        for i in range(20)
    ]

    req = RerankingRequest(
        query_id="QRY_BENCHMARK_001",
        destination="Bali",
        user_budget_idr=1200000.0,
        enable_exploration=True,
        hotels=sample_hotels,
    )

    res = engine.rerank(req)
    print(f"  -> Sample Reranking Result (Top 3 of 20):")
    for item in res.ranked_hotels[:3]:
        print(f"     #{item.rank} {item.hotel_id} | Stars: {item.star_rating} | Score: {item.debiased_score} | Badge: {item.merchandising_badge}")

    # Latency benchmark (100 runs)
    latencies = []
    for _ in range(100):
        r = engine.rerank(req)
        latencies.append(r.latency_ms)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"\n[SLA BENCHMARK] Real-Time Inference Latency (100 runs across 20 candidates):")
    print(f"  - Median (p50): {p50:.2f} ms")
    print(f"  - 95th Percentile (p95): {p95:.2f} ms")
    print(f"  - 99th Percentile (p99): {p99:.2f} ms")
    print(f"  -> Serving SLA (< 10ms target for p95): {'PASSED' if p95 < 10.0 else 'CHECK'} (p95 = {p95:.2f} ms)")

    print("\n" + "=" * 80)
    print(" [SUCCESS] ALL RANKCRAFT PIPELINE STEPS COMPLETED WITHOUT ERROR")
    print(" Ready for Production Deployment & Interview Presentation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
