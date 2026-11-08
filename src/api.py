"""
Real-Time Counterfactual Hotel Reranking Microservice (RankCraft)
Author: Ahmad Fathin Naufal (naufal1555)
Purpose:
  - Real-Time Search SERP reranking (< 10ms SLA target) for Traveloka Hotel Search.
  - Applies Doubly Robust counterfactual scoring to eliminate visual position bias.
  - Dynamically blends Bayesian Thompson Sampling for high-potential cold-start discovery.
"""

from typing import Dict, Any, List, Optional
import time
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from models import (
    RANKING_FEATURES,
    DoublyRobustRanker,
    IPSRanker,
    ContextualThompsonSamplingBandit,
)


app = FastAPI(
    title="RankCraft: Real-Time Counterfactual Reranking API",
    description="Debiased Search Ranking Engine with Contextual Thompson Sampling for Traveloka",
    version="1.0.0",
)


class HotelCandidate(BaseModel):
    hotel_id: str = Field(..., json_schema_extra={"example": "HTL_8842"})
    star_rating: int = Field(..., ge=1, le=5, json_schema_extra={"example": 4})
    review_score: float = Field(..., ge=1.0, le=10.0, json_schema_extra={"example": 8.8})
    distance_to_center_km: float = Field(..., ge=0.0, json_schema_extra={"example": 2.4})
    discount_pct: int = Field(0, ge=0, le=100, json_schema_extra={"example": 15})
    amenities_count: int = Field(..., ge=1, json_schema_extra={"example": 8})
    hotel_price_idr: float = Field(..., gt=0, json_schema_extra={"example": 950000.0})
    historical_ctr: float = Field(0.08, ge=0.0, le=1.0, json_schema_extra={"example": 0.08})
    impression_count: int = Field(100, ge=0, json_schema_extra={"example": 120})
    click_count: int = Field(10, ge=0, json_schema_extra={"example": 12})


class RerankingRequest(BaseModel):
    query_id: Optional[str] = Field(None, json_schema_extra={"example": "QRY_TVLK_BALI_01"})
    destination: str = Field(..., json_schema_extra={"example": "Bali"})
    user_budget_idr: float = Field(..., gt=0, json_schema_extra={"example": 1000000.0})
    enable_exploration: bool = Field(True, json_schema_extra={"example": True})
    hotels: List[HotelCandidate]


class RankedHotelItem(BaseModel):
    rank: int
    hotel_id: str
    debiased_score: float
    star_rating: int
    review_score: float
    hotel_price_idr: float
    is_exploration_pick: bool
    merchandising_badge: str


class RerankingResponse(BaseModel):
    query_id: str
    destination: str
    total_candidates: int
    ranking_method: str
    ranked_hotels: List[RankedHotelItem]
    latency_ms: float


class RankingServingEngine:
    def __init__(self):
        self.dr_ranker: Optional[DoublyRobustRanker] = None
        self.bandit: Optional[ContextualThompsonSamplingBandit] = None
        self.is_ready = False

    def load_models(
        self,
        dr_ranker: DoublyRobustRanker,
        bandit: Optional[ContextualThompsonSamplingBandit] = None,
    ):
        self.dr_ranker = dr_ranker
        self.bandit = bandit or ContextualThompsonSamplingBandit()
        self.is_ready = True

    def rerank(self, req: RerankingRequest) -> RerankingResponse:
        t0 = time.perf_counter()
        if not self.is_ready:
            raise RuntimeError("Serving Engine is not initialized with trained rankers.")

        q_id = req.query_id or f"QRY_TVLK_{uuid.uuid4().hex[:8]}"
        n = len(req.hotels)
        if n == 0:
            return RerankingResponse(
                query_id=q_id,
                destination=req.destination,
                total_candidates=0,
                ranking_method="None",
                ranked_hotels=[],
                latency_ms=0.0,
            )

        # Vectorized feature matrix construction (direct NumPy array)
        X = np.empty((n, len(RANKING_FEATURES)), dtype=np.float32)
        impr_counts = np.empty(n, dtype=int)
        click_counts = np.empty(n, dtype=int)

        for i, h in enumerate(req.hotels):
            affinity_ratio = round(h.hotel_price_idr / max(1.0, req.user_budget_idr), 2)
            X[i] = [
                h.star_rating,
                h.review_score,
                h.distance_to_center_km,
                h.discount_pct,
                h.amenities_count,
                h.hotel_price_idr,
                affinity_ratio,
                h.historical_ctr,
            ]
            impr_counts[i] = h.impression_count
            click_counts[i] = h.click_count

        # Score with Doubly Robust Ranker (using internal XGBRegressor directly)
        base_scores = self.dr_ranker.final_ranker.predict(X)

        # Contextual Bandit Exploration for Cold-Start
        if req.enable_exploration:
            final_scores, is_explored = self.bandit.rerank_with_exploration(
                base_scores=base_scores,
                impression_counts=impr_counts,
                click_counts=click_counts,
            )
        else:
            final_scores = base_scores
            is_explored = np.zeros(n, dtype=bool)

        # Sort in descending order
        rank_order = np.argsort(final_scores)[::-1]

        ranked_items = []
        for rank_pos, idx in enumerate(rank_order, start=1):
            h = req.hotels[idx]
            score_val = round(float(final_scores[idx]), 3)

            # Assign merchandising badge
            if rank_pos == 1:
                badge = "TOP_RECOMMENDED"
            elif is_explored[idx]:
                badge = "NEW_DISCOVERY"
            elif h.discount_pct >= 20:
                badge = "SPECIAL_DEAL"
            elif h.review_score >= 9.0:
                badge = "EXCEPTIONAL_RATING"
            else:
                badge = "BEST_VALUE"

            ranked_items.append(RankedHotelItem(
                rank=rank_pos,
                hotel_id=h.hotel_id,
                debiased_score=score_val,
                star_rating=h.star_rating,
                review_score=h.review_score,
                hotel_price_idr=h.hotel_price_idr,
                is_exploration_pick=bool(is_explored[idx]),
                merchandising_badge=badge,
            ))

        latency = round((time.perf_counter() - t0) * 1000.0, 3)

        return RerankingResponse(
            query_id=q_id,
            destination=req.destination,
            total_candidates=n,
            ranking_method="Doubly Robust CLTR + Contextual Thompson Sampling",
            ranked_hotels=ranked_items,
            latency_ms=latency,
        )


GLOBAL_RANK_ENGINE = RankingServingEngine()


@app.get("/v1/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "RankCraft Counterfactual Reranking API",
        "models_ready": GLOBAL_RANK_ENGINE.is_ready,
    }


@app.post("/v1/rank-hotels", response_model=RerankingResponse)
def rank_hotels(req: RerankingRequest):
    if not GLOBAL_RANK_ENGINE.is_ready:
        raise HTTPException(status_code=503, detail="Ranking models are not loaded.")
    try:
        return GLOBAL_RANK_ENGINE.rerank(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
