#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stock_agent_mvp.py

A single-file practical stock decision-support agent MVP.

Features:
- Market regime analysis
- Sector rotation analysis
- Stock scoring
- Risk control
- End-of-day reflection
- JSON + Markdown report output
- Mock mode by default
- Optional Yahoo Finance daily data mode

Run:
    python stock_agent_mvp.py

Optional live mode:
    pip install yfinance pandas numpy
    python stock_agent_mvp.py --live

Optional custom watchlist:
    python stock_agent_mvp.py --live --watchlist NVDA AMD AVGO TSM ASML MSFT META SMCI

Optional external config:
    python stock_agent_mvp.py --config config/strategy_config.json

Backtest (live dependencies required):
    python stock_agent_mvp.py --backtest --live
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

try:
    import yfinance as yf
except Exception:
    yf = None


# =========================
# Config
# =========================

DEFAULT_WATCHLIST = ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MSFT", "META", "SMCI"]
DEFAULT_INDICES = ["SPY", "QQQ", "DIA"]

SECTOR_PROXY = {
    "semiconductors": ["NVDA", "AMD", "AVGO", "TSM", "ASML"],
    "ai_infra": ["NVDA", "MSFT", "META", "SMCI"],
    "software_platform": ["MSFT", "META"],
    "broad_tech": ["QQQ"],
    "industrial": ["DIA"],
}

DEFAULT_APP_CONFIG = {
    "risk": {
        "max_total_position": 0.35,
        "max_single_position": 0.12,
        "stop_loss_pct_default": 0.04,
        "pause_after_consecutive_losses": 3,
        "forbid_gap_up_pct": 0.07,
        "min_stock_score": 70,
        "max_candidates": 5,
    },
    "sector_scoring": {
        "return_weight": 1.5,
        "volume_weight": 0.08,
        "leader_weight": 1.2,
        "breadth_weight": 10.0,
        "top_sector_min_score": 6.0,
        "avoid_sector_max_score": 2.0,
    },
    "market_regime": {
        "qqq_strong": 0.7,
        "qqq_weak": -0.7,
        "spy_strong": 0.4,
        "spy_weak": -0.4,
        "turnover_strong": 5.0,
        "turnover_weak": -5.0,
        "breadth_strong": 1.2,
        "breadth_weak": 0.8,
        "high_beta_damage_alert": 0.5,
        "volatility_alert": 3.0,
        "trend_strong_score": 2.2,
        "trend_moderate_score": 0.8,
        "risk_off_score": -0.8,
        "panic_score": -2.2,
    },
    "backtest": {
        "period": "12mo",
        "holding_days": 1,
        "warmup_bars": 30,
        "min_recommendations": 1,
    },
}

CONFIG_PATH = Path("config/strategy_config.json")


def deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_app_config(config_path: Path = CONFIG_PATH) -> Dict:
    if not config_path.exists():
        return DEFAULT_APP_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    return deep_merge(DEFAULT_APP_CONFIG, user_config)


APP_CONFIG = load_app_config()
RISK_CONFIG = APP_CONFIG["risk"]
SECTOR_SCORING_CONFIG = APP_CONFIG["sector_scoring"]
MARKET_REGIME_CONFIG = APP_CONFIG["market_regime"]
BACKTEST_CONFIG = APP_CONFIG["backtest"]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("stock_agent_mvp")


# =========================
# Data Models
# =========================

@dataclass
class MarketFeatures:
    index_returns: Dict[str, float]
    turnover_change_pct: float
    breadth_ratio: float
    volatility_pct: float
    breakout_count: int
    breakdown_count: int
    high_beta_damage: float


@dataclass
class SectorFeature:
    name: str
    return_pct: float
    volume_change_pct: float
    leaders_strength: float
    breadth: float
    catalyst_summary: str = ""


@dataclass
class StockFeatures:
    ticker: str
    close: float
    prev_close: float
    open_price: float
    high: float
    low: float
    ma5: float
    ma10: float
    ma20: float
    volume: float
    avg_volume_20: float
    volume_ratio: float
    breakout_score: float
    pullback_score: float
    sector: str
    sector_rank: int
    news_summary: str = ""
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class MarketRegimeOutput:
    market_regime: str
    confidence: float
    evidence: List[str]
    trading_permission: str
    warning_flags: List[str]


@dataclass
class SectorRotationOutput:
    top_sectors: List[Dict]
    avoid_sectors: List[Dict]
    comments: List[str]


@dataclass
class StockScoreOutput:
    ticker: str
    score_total: int
    score_breakdown: Dict[str, int]
    setup: str
    entry_zone: List[float]
    stop_loss: float
    take_profit_hint: List[float]
    risk_flags: List[str]
    rationale: List[str]


@dataclass
class RiskDecision:
    portfolio_risk_level: str
    max_total_position: float
    max_single_position: float
    forbidden_actions: List[str]
    emergency_rule: str


@dataclass
class ReflectionOutput:
    daily_reflection: List[str]
    strategy_adjustment: List[str]
    disabled_patterns: List[str]
    notes: List[str]


@dataclass
class Recommendation:
    ticker: str
    score: int
    setup: str
    entry_zone: List[float]
    stop_loss: float
    take_profit_hint: List[float]
    risk_flags: List[str]
    rationale: List[str]


@dataclass
class DailyPlan:
    date: str
    mode: str
    market_regime: str
    trading_permission: str
    top_sectors: List[Dict]
    recommendations: List[Recommendation]
    risk: Dict
    daily_no_go: List[str]
    reflection: Dict


@dataclass
class BacktestSummary:
    mode: str
    period: str
    days_tested: int
    days_with_signals: int
    total_picks: int
    win_rate: float
    avg_forward_return_pct: float
    cumulative_return_pct: float
    max_drawdown_pct: float


# =========================
# Utilities
# =========================

def round2(x: float) -> float:
    return round(float(x), 2)


def pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def sma(values: List[float], window: int) -> float:
    if len(values) < window:
        return sum(values) / max(len(values), 1)
    slice_ = values[-window:]
    return sum(slice_) / window


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# =========================
# Mock Data
# =========================

def load_mock_market() -> MarketFeatures:
    return MarketFeatures(
        index_returns={"SPY": 0.62, "QQQ": 0.94, "DIA": 0.18},
        turnover_change_pct=8.7,
        breadth_ratio=1.42,
        volatility_pct=1.8,
        breakout_count=74,
        breakdown_count=21,
        high_beta_damage=0.35,
    )


def load_mock_sectors() -> List[SectorFeature]:
    return [
        SectorFeature(
            name="semiconductors",
            return_pct=2.3,
            volume_change_pct=18.5,
            leaders_strength=8.9,
            breadth=0.71,
            catalyst_summary="earnings optimism and AI demand",
        ),
        SectorFeature(
            name="ai_infra",
            return_pct=1.9,
            volume_change_pct=14.1,
            leaders_strength=8.2,
            breadth=0.68,
            catalyst_summary="cloud capex expansion",
        ),
        SectorFeature(
            name="software_platform",
            return_pct=0.7,
            volume_change_pct=3.6,
            leaders_strength=6.1,
            breadth=0.55,
            catalyst_summary="steady large-cap support",
        ),
        SectorFeature(
            name="industrial",
            return_pct=-0.4,
            volume_change_pct=-3.2,
            leaders_strength=3.8,
            breadth=0.42,
            catalyst_summary="defensive rotation fading",
        ),
    ]


def load_mock_stocks() -> List[StockFeatures]:
    return [
        StockFeatures(
            ticker="NVDA",
            close=121.3,
            prev_close=119.9,
            open_price=120.2,
            high=122.4,
            low=119.8,
            ma5=119.0,
            ma10=116.2,
            ma20=112.8,
            volume=120_000_000,
            avg_volume_20=70_000_000,
            volume_ratio=1.71,
            breakout_score=8.8,
            pullback_score=6.1,
            sector="semiconductors",
            sector_rank=1,
            news_summary="strong demand narrative",
        ),
        StockFeatures(
            ticker="AMD",
            close=178.6,
            prev_close=176.5,
            open_price=177.0,
            high=179.2,
            low=175.8,
            ma5=176.4,
            ma10=172.1,
            ma20=169.9,
            volume=65_000_000,
            avg_volume_20=50_000_000,
            volume_ratio=1.30,
            breakout_score=7.2,
            pullback_score=7.8,
            sector="semiconductors",
            sector_rank=1,
            news_summary="product pipeline support",
        ),
        StockFeatures(
            ticker="ASML",
            close=1032.4,
            prev_close=1020.0,
            open_price=1025.0,
            high=1038.5,
            low=1018.2,
            ma5=1010.8,
            ma10=998.0,
            ma20=985.4,
            volume=1_800_000,
            avg_volume_20=1_560_000,
            volume_ratio=1.15,
            breakout_score=6.9,
            pullback_score=8.1,
            sector="semiconductors",
            sector_rank=1,
            news_summary="equipment leadership intact",
        ),
        StockFeatures(
            ticker="MSFT",
            close=432.2,
            prev_close=430.8,
            open_price=431.0,
            high=433.7,
            low=428.9,
            ma5=429.8,
            ma10=425.5,
            ma20=420.2,
            volume=25_000_000,
            avg_volume_20=23_000_000,
            volume_ratio=1.09,
            breakout_score=5.8,
            pullback_score=7.6,
            sector="ai_infra",
            sector_rank=2,
            news_summary="cloud capex resilience",
        ),
        StockFeatures(
            ticker="META",
            close=618.4,
            prev_close=615.2,
            open_price=616.0,
            high=620.1,
            low=611.8,
            ma5=614.0,
            ma10=607.8,
            ma20=598.5,
            volume=16_000_000,
            avg_volume_20=14_000_000,
            volume_ratio=1.14,
            breakout_score=6.5,
            pullback_score=7.1,
            sector="ai_infra",
            sector_rank=2,
            news_summary="AI monetization tailwind",
        ),
    ]


# =========================
# Yahoo Finance Live Data
# =========================

def require_live_dependencies() -> None:
    if pd is None or np is None or yf is None:
        raise RuntimeError(
            "Live mode requires pandas, numpy, and yfinance. "
            "Install with: pip install pandas numpy yfinance"
        )


def download_ohlcv(tickers: List[str], period: str = "3mo", interval: str = "1d") -> Dict[str, "pd.DataFrame"]:
    require_live_dependencies()
    result: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=False, progress=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna(how="all")
            result[t] = df.copy()
        except Exception as e:
            logger.warning("Failed to download %s: %s", t, e)
            continue
    return result


def infer_sector_for_ticker(ticker: str) -> str:
    for sector, names in SECTOR_PROXY.items():
        if ticker in names:
            return sector
    return "broad_tech"


def build_live_market_features(index_data: Dict[str, "pd.DataFrame"], stock_data: Dict[str, "pd.DataFrame"]) -> MarketFeatures:
    index_returns = {}
    turnover_changes = []
    breakout_count = 0
    breakdown_count = 0
    high_beta_damage = 0.0

    all_last_returns = []

    for ticker, df in index_data.items():
        if len(df) < 2:
            index_returns[ticker] = 0.0
            continue
        close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        index_returns[ticker] = round2((close / prev_close - 1) * 100.0)

        if "Volume" in df.columns and len(df) >= 21:
            vol = float(df["Volume"].iloc[-1])
            avg20 = float(df["Volume"].iloc[-21:-1].mean())
            turnover_changes.append(pct_change(vol, avg20))

    stock_up = 0
    stock_down = 0

    for ticker, df in stock_data.items():
        if len(df) < 21:
            continue

        close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        ma20 = float(df["Close"].iloc[-20:].mean())
        last_return = (close / prev_close - 1) * 100.0
        all_last_returns.append(abs(last_return))

        if close > prev_close:
            stock_up += 1
        else:
            stock_down += 1

        prev_20_high = float(df["High"].iloc[-21:-1].max())
        prev_20_low = float(df["Low"].iloc[-21:-1].min())

        if close > prev_20_high:
            breakout_count += 1
        if close < prev_20_low:
            breakdown_count += 1

        if ticker in {"NVDA", "AMD", "SMCI", "META"}:
            if close < ma20:
                high_beta_damage += 1.0

    breadth_ratio = stock_up / max(stock_down, 1)
    volatility_pct = float(np.mean(all_last_returns)) if all_last_returns else 1.0

    turnover_change_pct = round2(sum(turnover_changes) / max(len(turnover_changes), 1))
    high_beta_damage = round2(high_beta_damage / max(1, len([t for t in stock_data if t in {"NVDA", "AMD", "SMCI", "META"}])))

    return MarketFeatures(
        index_returns=index_returns,
        turnover_change_pct=turnover_change_pct,
        breadth_ratio=round2(breadth_ratio),
        volatility_pct=round2(volatility_pct),
        breakout_count=breakout_count,
        breakdown_count=breakdown_count,
        high_beta_damage=high_beta_damage,
    )


def build_live_sector_features(stock_data: Dict[str, "pd.DataFrame"]) -> List[SectorFeature]:
    sector_rows = []

    for sector, tickers in SECTOR_PROXY.items():
        returns = []
        vol_changes = []
        leader_strengths = []
        breadth_pass = 0
        valid = 0

        for t in tickers:
            df = stock_data.get(t)
            if df is None or len(df) < 21:
                continue

            valid += 1
            close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            ma20 = float(df["Close"].iloc[-20:].mean())
            ret = (close / prev_close - 1) * 100.0
            returns.append(ret)

            vol = float(df["Volume"].iloc[-1])
            avg20 = float(df["Volume"].iloc[-21:-1].mean())
            vol_changes.append(pct_change(vol, avg20))

            dist_to_ma20 = ((close / ma20) - 1) * 100.0
            leader_strengths.append(dist_to_ma20)

            if close > ma20:
                breadth_pass += 1

        if valid == 0:
            continue

        sector_rows.append(
            SectorFeature(
                name=sector,
                return_pct=round2(sum(returns) / len(returns)),
                volume_change_pct=round2(sum(vol_changes) / len(vol_changes)),
                leaders_strength=round2(sum(leader_strengths) / len(leader_strengths)),
                breadth=round2(breadth_pass / valid),
                catalyst_summary="live price/volume strength proxy",
            )
        )

    return sector_rows


def build_live_stock_features(watchlist: List[str], stock_data: Dict[str, "pd.DataFrame"], sector_ranks: Dict[str, int]) -> List[StockFeatures]:
    rows = []

    for ticker in watchlist:
        df = stock_data.get(ticker)
        if df is None or len(df) < 21:
            continue

        close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        open_price = float(df["Open"].iloc[-1])
        high = float(df["High"].iloc[-1])
        low = float(df["Low"].iloc[-1])

        closes = df["Close"].tolist()
        ma5 = sma(closes, 5)
        ma10 = sma(closes, 10)
        ma20 = sma(closes, 20)

        volume = float(df["Volume"].iloc[-1])
        avg_volume_20 = float(df["Volume"].iloc[-21:-1].mean()) if len(df) >= 21 else volume
        volume_ratio = volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

        prev_20_high = float(df["High"].iloc[-21:-1].max())
        breakout_raw = ((close / prev_20_high) - 1) * 100.0
        breakout_score = clamp(5 + breakout_raw * 2, 0, 10)

        dist_to_ma10 = abs((close / ma10 - 1) * 100.0) if ma10 else 0
        pullback_score = clamp(10 - dist_to_ma10, 0, 10)

        sector = infer_sector_for_ticker(ticker)
        risk_flags = []

        gap_pct = (open_price / prev_close - 1) if prev_close > 0 else 0
        if gap_pct > RISK_CONFIG["forbid_gap_up_pct"]:
            risk_flags.append(f"gap_up_{round2(gap_pct * 100)}pct")

        if close < ma20:
            risk_flags.append("below_ma20")

        if high > 0 and (high - close) / high > 0.03:
            risk_flags.append("possible_intraday_reversal")

        rows.append(
            StockFeatures(
                ticker=ticker,
                close=round2(close),
                prev_close=round2(prev_close),
                open_price=round2(open_price),
                high=round2(high),
                low=round2(low),
                ma5=round2(ma5),
                ma10=round2(ma10),
                ma20=round2(ma20),
                volume=round2(volume),
                avg_volume_20=round2(avg_volume_20),
                volume_ratio=round2(volume_ratio),
                breakout_score=round2(breakout_score),
                pullback_score=round2(pullback_score),
                sector=sector,
                sector_rank=sector_ranks.get(sector, 99),
                news_summary="live market structure only; no news adapter yet",
                risk_flags=risk_flags,
            )
        )

    return rows


# =========================
# Agents
# =========================

class MarketRegimeAgent:
    def run(self, market: MarketFeatures) -> MarketRegimeOutput:
        score = 0.0
        evidence = []
        warning_flags = []

        qqq_ret = market.index_returns.get("QQQ", 0.0)
        spy_ret = market.index_returns.get("SPY", 0.0)

        if qqq_ret > MARKET_REGIME_CONFIG["qqq_strong"]:
            score += 1.2
            evidence.append(f"QQQ strong at {qqq_ret:.2f}%")
        elif qqq_ret < MARKET_REGIME_CONFIG["qqq_weak"]:
            score -= 1.2
            evidence.append(f"QQQ weak at {qqq_ret:.2f}%")

        if spy_ret > MARKET_REGIME_CONFIG["spy_strong"]:
            score += 0.8
            evidence.append(f"SPY supportive at {spy_ret:.2f}%")
        elif spy_ret < MARKET_REGIME_CONFIG["spy_weak"]:
            score -= 0.8
            evidence.append(f"SPY soft at {spy_ret:.2f}%")

        if market.turnover_change_pct > MARKET_REGIME_CONFIG["turnover_strong"]:
            score += 0.8
            evidence.append(f"turnover expanding {market.turnover_change_pct:.2f}%")
        elif market.turnover_change_pct < MARKET_REGIME_CONFIG["turnover_weak"]:
            score -= 0.8
            evidence.append(f"turnover contracting {market.turnover_change_pct:.2f}%")

        if market.breadth_ratio > MARKET_REGIME_CONFIG["breadth_strong"]:
            score += 0.8
            evidence.append(f"breadth healthy at {market.breadth_ratio:.2f}")
        elif market.breadth_ratio < MARKET_REGIME_CONFIG["breadth_weak"]:
            score -= 0.8
            evidence.append(f"breadth weak at {market.breadth_ratio:.2f}")

        if market.breakout_count > market.breakdown_count * 1.5:
            score += 0.8
            evidence.append("breakouts dominate breakdowns")
        elif market.breakdown_count > market.breakout_count * 1.2:
            score -= 0.8
            evidence.append("breakdowns dominate breakouts")

        if market.high_beta_damage > MARKET_REGIME_CONFIG["high_beta_damage_alert"]:
            score -= 0.8
            warning_flags.append("high_beta_damage_rising")
            evidence.append(f"high beta damage elevated at {market.high_beta_damage:.2f}")

        if market.volatility_pct > MARKET_REGIME_CONFIG["volatility_alert"]:
            score -= 0.4
            warning_flags.append("market_volatility_elevated")
            evidence.append(f"volatility elevated at {market.volatility_pct:.2f}%")

        if score >= MARKET_REGIME_CONFIG["trend_strong_score"]:
            regime = "trend_strong"
            permission = "normal_to_aggressive"
        elif score >= MARKET_REGIME_CONFIG["trend_moderate_score"]:
            regime = "trend_moderate"
            permission = "normal"
        elif score > MARKET_REGIME_CONFIG["risk_off_score"]:
            regime = "range_choppy"
            permission = "light_only"
            warning_flags.append("prefer_pullbacks_over_chasing")
        elif score > MARKET_REGIME_CONFIG["panic_score"]:
            regime = "risk_off"
            permission = "defensive_only"
            warning_flags.append("avoid_aggressive_entries")
        else:
            regime = "panic_distribution"
            permission = "no_new_risk"
            warning_flags.append("capital_preservation_mode")

        confidence = clamp(abs(score) / 3.0, 0.45, 0.95)

        return MarketRegimeOutput(
            market_regime=regime,
            confidence=round2(confidence),
            evidence=evidence,
            trading_permission=permission,
            warning_flags=warning_flags,
        )


class SectorRotationAgent:
    def run(self, sectors: List[SectorFeature]) -> SectorRotationOutput:
        scored = []

        for s in sectors:
            strength = (
                s.return_pct * SECTOR_SCORING_CONFIG["return_weight"]
                + s.volume_change_pct * SECTOR_SCORING_CONFIG["volume_weight"]
                + s.leaders_strength * SECTOR_SCORING_CONFIG["leader_weight"]
                + s.breadth * SECTOR_SCORING_CONFIG["breadth_weight"]
            )
            scored.append((s, strength))

        scored.sort(key=lambda x: x[1], reverse=True)

        top_sectors = []
        avoid_sectors = []
        comments = []

        for rank, (sector, score) in enumerate(scored, start=1):
            item = {
                "name": sector.name,
                "rank": rank,
                "strength_score": round2(score),
                "stage": infer_sector_stage(sector),
                "return_pct": sector.return_pct,
                "volume_change_pct": sector.volume_change_pct,
                "breadth": sector.breadth,
                "catalyst_summary": sector.catalyst_summary,
            }
            if rank <= 3 and score > SECTOR_SCORING_CONFIG["top_sector_min_score"]:
                top_sectors.append(item)
            elif score < SECTOR_SCORING_CONFIG["avoid_sector_max_score"]:
                avoid_sectors.append(
                    {
                        "name": sector.name,
                        "reason": f"weak structure: score={round2(score)}",
                    }
                )

        if top_sectors:
            comments.append("Focus on strongest sectors first; avoid random diversification.")
        if avoid_sectors:
            comments.append("Weak sectors should be deprioritized unless new catalysts appear.")

        return SectorRotationOutput(
            top_sectors=top_sectors,
            avoid_sectors=avoid_sectors,
            comments=comments,
        )


def infer_sector_stage(sector: SectorFeature) -> str:
    if sector.return_pct > 1.5 and sector.volume_change_pct > 8 and sector.breadth > 0.6:
        return "strengthening"
    if sector.return_pct > 0.5 and sector.breadth > 0.5:
        return "constructive"
    if sector.return_pct < -0.5 and sector.breadth < 0.45:
        return "weakening"
    return "mixed"


class StockScoringAgent:
    def run(self, stock: StockFeatures, market: MarketRegimeOutput) -> StockScoreOutput:
        score_breakdown = numeric_stock_score(stock, market)
        total = sum(score_breakdown.values())

        setup = infer_setup(stock, market)

        entry_zone = infer_entry_zone(stock, setup)
        stop_loss = infer_stop_loss(stock, setup)
        take_profit_hint = infer_take_profit(stock, stop_loss)

        risk_flags = list(stock.risk_flags)
        rationale = []

        if stock.ma5 > stock.ma10 > stock.ma20:
            rationale.append("moving averages aligned bullishly")
        if stock.volume_ratio >= 1.2:
            rationale.append(f"volume confirmation at {stock.volume_ratio:.2f}x")
        if stock.sector_rank <= 2:
            rationale.append(f"belongs to top-ranked sector: {stock.sector}")
        if stock.breakout_score >= 7.0:
            rationale.append("price structure near or above breakout zone")
        if setup == "pullback_buy":
            rationale.append("better treated as controlled pullback entry rather than chase")

        if stock.close < stock.ma20:
            risk_flags.append("trend_not_fully_confirmed")
        if stock.open_price > stock.prev_close * (1 + RISK_CONFIG["forbid_gap_up_pct"]):
            risk_flags.append("do_not_chase_large_gap_up")
        if market.trading_permission in {"light_only", "defensive_only", "no_new_risk"}:
            risk_flags.append("market_environment_requires_restraint")

        return StockScoreOutput(
            ticker=stock.ticker,
            score_total=int(total),
            score_breakdown=score_breakdown,
            setup=setup,
            entry_zone=[round2(entry_zone[0]), round2(entry_zone[1])],
            stop_loss=round2(stop_loss),
            take_profit_hint=[round2(take_profit_hint[0]), round2(take_profit_hint[1])],
            risk_flags=sorted(set(risk_flags)),
            rationale=rationale,
        )


def numeric_stock_score(stock: StockFeatures, market: MarketRegimeOutput) -> Dict[str, int]:
    trend = 0
    flow = 0
    execution = 0
    sector = 0

    if stock.ma5 > stock.ma10 > stock.ma20:
        trend += 22
    elif stock.ma10 > stock.ma20:
        trend += 14
    elif stock.close > stock.ma20:
        trend += 8

    trend += min(int(stock.breakout_score * 1.5), 15)

    flow += min(int(stock.volume_ratio * 10), 18)

    distance_to_ma10 = abs((stock.close / stock.ma10 - 1) * 100.0) if stock.ma10 else 0
    execution += int(clamp(15 - distance_to_ma10 * 2, 0, 15))
    execution += min(int(stock.pullback_score), 10)

    sector = max(0, 12 - stock.sector_rank * 2)

    if market.market_regime == "trend_strong":
        trend += 4
    elif market.market_regime == "range_choppy":
        execution += 2
    elif market.market_regime in {"risk_off", "panic_distribution"}:
        trend -= 4
        execution -= 2

    trend = max(trend, 0)
    flow = max(flow, 0)
    execution = max(execution, 0)
    sector = max(sector, 0)

    return {
        "trend": int(trend),
        "flow": int(flow),
        "execution": int(execution),
        "sector": int(sector),
    }


def infer_setup(stock: StockFeatures, market: MarketRegimeOutput) -> str:
    if market.market_regime == "trend_strong" and stock.breakout_score >= 7.5 and stock.volume_ratio >= 1.2:
        return "trend_continuation"
    if stock.pullback_score >= 7.0 and stock.close >= stock.ma10:
        return "pullback_buy"
    if stock.breakout_score >= 6.5:
        return "early_breakout_watch"
    return "watch_only"


def infer_entry_zone(stock: StockFeatures, setup: str) -> Tuple[float, float]:
    if setup == "trend_continuation":
        low = min(stock.close, stock.high * 0.995)
        high = stock.close * 1.01
    elif setup == "pullback_buy":
        anchor = max(stock.ma5, stock.ma10)
        low = anchor * 0.995
        high = anchor * 1.01
    elif setup == "early_breakout_watch":
        low = stock.close * 0.99
        high = stock.close * 1.005
    else:
        low = stock.close * 0.97
        high = stock.close * 0.99
    return low, high


def infer_stop_loss(stock: StockFeatures, setup: str) -> float:
    if setup == "trend_continuation":
        return min(stock.ma10, stock.close * (1 - RISK_CONFIG["stop_loss_pct_default"]))
    if setup == "pullback_buy":
        return min(stock.ma20, stock.ma10 * 0.985)
    if setup == "early_breakout_watch":
        return stock.close * 0.96
    return stock.close * 0.95


def infer_take_profit(stock: StockFeatures, stop_loss: float) -> Tuple[float, float]:
    risk_per_share = max(stock.close - stop_loss, stock.close * 0.02)
    tp1 = stock.close + risk_per_share * 1.5
    tp2 = stock.close + risk_per_share * 2.5
    return tp1, tp2


class RiskAgent:
    def __init__(self, config: Dict):
        self.config = config

    def run(self, market: MarketRegimeOutput, consecutive_losses: int = 0) -> RiskDecision:
        base_total = self.config["max_total_position"]
        base_single = self.config["max_single_position"]

        forbidden_actions = []
        risk_level = "medium"
        emergency_rule = "Any position breaching stop loss must be exited."

        if market.market_regime in {"risk_off", "panic_distribution"}:
            base_total = min(base_total, 0.20)
            base_single = min(base_single, 0.08)
            forbidden_actions.append("Avoid aggressive breakout chasing.")
            risk_level = "high"

        if market.market_regime == "range_choppy":
            base_total = min(base_total, 0.25)
            base_single = min(base_single, 0.10)
            forbidden_actions.append("Avoid oversized positions in choppy markets.")
            risk_level = "medium_high"

        if market.market_regime == "trend_strong":
            risk_level = "medium"

        if consecutive_losses >= self.config["pause_after_consecutive_losses"]:
            base_total = 0.0
            base_single = 0.0
            forbidden_actions.append("Pause new entries after repeated losses.")
            risk_level = "very_high"

        forbidden_actions.append(
            f"Do not chase stocks gapping above {self.config['forbid_gap_up_pct']:.0%}."
        )

        return RiskDecision(
            portfolio_risk_level=risk_level,
            max_total_position=round2(base_total),
            max_single_position=round2(base_single),
            forbidden_actions=forbidden_actions,
            emergency_rule=emergency_rule,
        )


class ReflectionAgent:
    def run(
        self,
        market: MarketRegimeOutput,
        sectors: SectorRotationOutput,
        recommendations: List[StockScoreOutput],
    ) -> ReflectionOutput:
        daily_reflection = []
        strategy_adjustment = []
        disabled_patterns = []
        notes = []

        if market.market_regime == "trend_strong":
            daily_reflection.append("Market backdrop supports trend-following entries better than random mean reversion.")
            strategy_adjustment.append("Prefer leaders in top sectors and buy controlled continuation setups.")
        elif market.market_regime == "range_choppy":
            daily_reflection.append("Market appears choppy; pullback entries are safer than breakout chasing.")
            strategy_adjustment.append("Reduce size and tighten selectivity.")
            disabled_patterns.append("aggressive_high_open_chasing")
        elif market.market_regime in {"risk_off", "panic_distribution"}:
            daily_reflection.append("Capital preservation matters more than idea count in weak tape.")
            strategy_adjustment.append("Only keep observation list or very small pilot positions.")
            disabled_patterns.append("new_breakout_entries")

        if sectors.top_sectors:
            sector_names = ", ".join(s["name"] for s in sectors.top_sectors[:3])
            notes.append(f"Top areas today: {sector_names}")

        rich_risk_flags = sum(1 for r in recommendations if len(r.risk_flags) >= 2)
        if rich_risk_flags >= max(1, len(recommendations) // 2):
            notes.append("Many candidates still carry non-trivial risk flags; execution discipline is important.")

        if not recommendations:
            daily_reflection.append("No stocks passed the threshold; that itself is valid information.")
            strategy_adjustment.append("Stay selective instead of forcing trades.")

        return ReflectionOutput(
            daily_reflection=daily_reflection,
            strategy_adjustment=strategy_adjustment,
            disabled_patterns=disabled_patterns,
            notes=notes,
        )


# =========================
# Pipeline
# =========================

def build_mock_inputs() -> Tuple[MarketFeatures, List[SectorFeature], List[StockFeatures]]:
    market = load_mock_market()
    sectors = load_mock_sectors()
    stocks = load_mock_stocks()
    return market, sectors, stocks


def build_live_inputs(watchlist: List[str]) -> Tuple[MarketFeatures, List[SectorFeature], List[StockFeatures]]:
    require_live_dependencies()
    tickers = sorted(set(DEFAULT_INDICES + watchlist + [x for v in SECTOR_PROXY.values() for x in v]))
    data = download_ohlcv(tickers)

    index_data = {k: v for k, v in data.items() if k in DEFAULT_INDICES}
    stock_data = {k: v for k, v in data.items() if k not in DEFAULT_INDICES}

    market = build_live_market_features(index_data, stock_data)
    sectors = build_live_sector_features(stock_data)

    sector_scores = []
    for s in sectors:
        score = (
            s.return_pct * SECTOR_SCORING_CONFIG["return_weight"]
            + s.volume_change_pct * SECTOR_SCORING_CONFIG["volume_weight"]
            + s.leaders_strength * SECTOR_SCORING_CONFIG["leader_weight"]
            + s.breadth * SECTOR_SCORING_CONFIG["breadth_weight"]
        )
        sector_scores.append((s.name, score))
    sector_scores.sort(key=lambda x: x[1], reverse=True)
    sector_ranks = {name: idx + 1 for idx, (name, _) in enumerate(sector_scores)}

    stocks = build_live_stock_features(watchlist, stock_data, sector_ranks)
    return market, sectors, stocks


def build_daily_plan(mode: str, watchlist: List[str], consecutive_losses: int = 0) -> DailyPlan:
    if mode == "live":
        market, sectors, stocks = build_live_inputs(watchlist)
    else:
        market, sectors, stocks = build_mock_inputs()

    market_agent = MarketRegimeAgent()
    sector_agent = SectorRotationAgent()
    stock_agent = StockScoringAgent()
    risk_agent = RiskAgent(RISK_CONFIG)
    reflection_agent = ReflectionAgent()

    market_out = market_agent.run(market)
    sector_out = sector_agent.run(sectors)

    scored = [stock_agent.run(s, market_out) for s in stocks]
    scored = sorted(scored, key=lambda x: x.score_total, reverse=True)
    scored = [s for s in scored if s.score_total >= RISK_CONFIG["min_stock_score"]]
    scored = scored[: RISK_CONFIG["max_candidates"]]

    risk = risk_agent.run(market_out, consecutive_losses=consecutive_losses)
    reflection = reflection_agent.run(market_out, sector_out, scored)

    recommendations = [
        Recommendation(
            ticker=s.ticker,
            score=s.score_total,
            setup=s.setup,
            entry_zone=s.entry_zone,
            stop_loss=s.stop_loss,
            take_profit_hint=s.take_profit_hint,
            risk_flags=s.risk_flags,
            rationale=s.rationale,
        )
        for s in scored
    ]

    daily_no_go = list(risk.forbidden_actions) + list(market_out.warning_flags)

    return DailyPlan(
        date=today_str(),
        mode=mode,
        market_regime=market_out.market_regime,
        trading_permission=market_out.trading_permission,
        top_sectors=sector_out.top_sectors,
        recommendations=recommendations,
        risk=asdict(risk),
        daily_no_go=daily_no_go,
        reflection=asdict(reflection),
    )


def build_stock_features_from_history(watchlist: List[str], stock_data: Dict[str, "pd.DataFrame"], sector_ranks: Dict[str, int], end_idx: int) -> List[StockFeatures]:
    rows = []
    for ticker in watchlist:
        df = stock_data.get(ticker)
        if df is None or len(df) <= end_idx or end_idx < 20:
            continue
        hist = df.iloc[: end_idx + 1]
        close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])
        open_price = float(hist["Open"].iloc[-1])
        high = float(hist["High"].iloc[-1])
        low = float(hist["Low"].iloc[-1])
        closes = hist["Close"].tolist()
        ma5 = sma(closes, 5)
        ma10 = sma(closes, 10)
        ma20 = sma(closes, 20)
        volume = float(hist["Volume"].iloc[-1])
        avg_volume_20 = float(hist["Volume"].iloc[-21:-1].mean())
        volume_ratio = volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        prev_20_high = float(hist["High"].iloc[-21:-1].max())
        breakout_raw = ((close / prev_20_high) - 1) * 100.0
        breakout_score = clamp(5 + breakout_raw * 2, 0, 10)
        dist_to_ma10 = abs((close / ma10 - 1) * 100.0) if ma10 else 0
        pullback_score = clamp(10 - dist_to_ma10, 0, 10)
        sector = infer_sector_for_ticker(ticker)
        risk_flags = []
        gap_pct = (open_price / prev_close - 1) if prev_close > 0 else 0
        if gap_pct > RISK_CONFIG["forbid_gap_up_pct"]:
            risk_flags.append(f"gap_up_{round2(gap_pct * 100)}pct")
        if close < ma20:
            risk_flags.append("below_ma20")
        rows.append(
            StockFeatures(
                ticker=ticker, close=round2(close), prev_close=round2(prev_close), open_price=round2(open_price),
                high=round2(high), low=round2(low), ma5=round2(ma5), ma10=round2(ma10), ma20=round2(ma20),
                volume=round2(volume), avg_volume_20=round2(avg_volume_20), volume_ratio=round2(volume_ratio),
                breakout_score=round2(breakout_score), pullback_score=round2(pullback_score), sector=sector,
                sector_rank=sector_ranks.get(sector, 99), news_summary="backtest mode", risk_flags=risk_flags,
            )
        )
    return rows


def run_backtest(watchlist: List[str], consecutive_losses: int = 0) -> BacktestSummary:
    require_live_dependencies()
    tickers = sorted(set(DEFAULT_INDICES + watchlist + [x for v in SECTOR_PROXY.values() for x in v]))
    data = download_ohlcv(tickers, period=BACKTEST_CONFIG["period"], interval="1d")
    index_data = {k: v for k, v in data.items() if k in DEFAULT_INDICES}
    stock_data = {k: v for k, v in data.items() if k not in DEFAULT_INDICES}

    all_lengths = [len(df) for df in stock_data.values() if len(df) > BACKTEST_CONFIG["warmup_bars"] + BACKTEST_CONFIG["holding_days"]]
    if not all_lengths:
        raise RuntimeError("Not enough data for backtest.")
    max_bars = min(all_lengths)
    holding_days = int(BACKTEST_CONFIG["holding_days"])
    warmup = int(BACKTEST_CONFIG["warmup_bars"])

    forward_returns = []
    daily_curve = [0.0]
    days_with_signals = 0

    market_agent = MarketRegimeAgent()
    sector_agent = SectorRotationAgent()
    stock_agent = StockScoringAgent()
    risk_agent = RiskAgent(RISK_CONFIG)

    for end_idx in range(warmup, max_bars - holding_days):
        hist_index = {k: v.iloc[: end_idx + 1] for k, v in index_data.items() if len(v) > end_idx}
        hist_stock = {k: v.iloc[: end_idx + 1] for k, v in stock_data.items() if len(v) > end_idx}
        market = build_live_market_features(hist_index, hist_stock)
        sectors = build_live_sector_features(hist_stock)
        if not sectors:
            continue
        sector_scores = []
        for s in sectors:
            score = (
                s.return_pct * SECTOR_SCORING_CONFIG["return_weight"]
                + s.volume_change_pct * SECTOR_SCORING_CONFIG["volume_weight"]
                + s.leaders_strength * SECTOR_SCORING_CONFIG["leader_weight"]
                + s.breadth * SECTOR_SCORING_CONFIG["breadth_weight"]
            )
            sector_scores.append((s.name, score))
        sector_scores.sort(key=lambda x: x[1], reverse=True)
        sector_ranks = {name: idx + 1 for idx, (name, _) in enumerate(sector_scores)}
        stocks = build_stock_features_from_history(watchlist, stock_data, sector_ranks, end_idx)

        market_out = market_agent.run(market)
        _ = risk_agent.run(market_out, consecutive_losses=consecutive_losses)
        scored = [stock_agent.run(s, market_out) for s in stocks]
        scored = sorted(scored, key=lambda x: x.score_total, reverse=True)
        scored = [s for s in scored if s.score_total >= RISK_CONFIG["min_stock_score"]][: RISK_CONFIG["max_candidates"]]

        if len(scored) < BACKTEST_CONFIG["min_recommendations"]:
            continue

        days_with_signals += 1
        period_returns = []
        for rec in scored:
            df = stock_data.get(rec.ticker)
            if df is None or len(df) <= end_idx + holding_days:
                continue
            entry = float(df["Close"].iloc[end_idx])
            exit_ = float(df["Close"].iloc[end_idx + holding_days])
            period_returns.append((exit_ / entry - 1) * 100.0)
        if not period_returns:
            continue
        day_ret = sum(period_returns) / len(period_returns)
        forward_returns.append(day_ret)
        daily_curve.append(daily_curve[-1] + day_ret)

    if not forward_returns:
        raise RuntimeError("Backtest produced no valid signals.")

    wins = sum(1 for r in forward_returns if r > 0)
    peak = daily_curve[0]
    max_dd = 0.0
    for x in daily_curve:
        peak = max(peak, x)
        max_dd = min(max_dd, x - peak)

    summary = BacktestSummary(
        mode="live_backtest",
        period=BACKTEST_CONFIG["period"],
        days_tested=len(forward_returns),
        days_with_signals=days_with_signals,
        total_picks=days_with_signals * RISK_CONFIG["max_candidates"],
        win_rate=round2(wins / len(forward_returns) * 100.0),
        avg_forward_return_pct=round2(sum(forward_returns) / len(forward_returns)),
        cumulative_return_pct=round2(sum(forward_returns)),
        max_drawdown_pct=round2(max_dd),
    )

    path = OUTPUT_DIR / f"backtest_summary_{today_str()}.json"
    path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Backtest summary written to: %s", path)
    return summary


# =========================
# Reporting
# =========================

def plan_to_dict(plan: DailyPlan) -> Dict:
    return {
        "date": plan.date,
        "mode": plan.mode,
        "market_regime": plan.market_regime,
        "trading_permission": plan.trading_permission,
        "top_sectors": plan.top_sectors,
        "recommendations": [asdict(r) for r in plan.recommendations],
        "risk": plan.risk,
        "daily_no_go": plan.daily_no_go,
        "reflection": plan.reflection,
    }


def write_json_report(plan: DailyPlan) -> Path:
    path = OUTPUT_DIR / f"daily_plan_{plan.date}.json"
    path.write_text(json.dumps(plan_to_dict(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_markdown_report(plan: DailyPlan) -> Path:
    path = OUTPUT_DIR / f"daily_plan_{plan.date}.md"

    lines = []
    lines.append(f"# Daily Plan — {plan.date}")
    lines.append("")
    lines.append(f"- Mode: `{plan.mode}`")
    lines.append(f"- Market regime: **{plan.market_regime}**")
    lines.append(f"- Trading permission: **{plan.trading_permission}**")
    lines.append("")

    lines.append("## Top Sectors")
    if plan.top_sectors:
        for s in plan.top_sectors:
            lines.append(
                f"- **{s['name']}** | rank={s['rank']} | strength={s['strength_score']} | "
                f"stage={s['stage']} | return={s['return_pct']}%"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Recommendations")
    if plan.recommendations:
        for r in plan.recommendations:
            lines.append(f"### {r.ticker}")
            lines.append(f"- Score: **{r.score}**")
            lines.append(f"- Setup: `{r.setup}`")
            lines.append(f"- Entry zone: `{r.entry_zone[0]} - {r.entry_zone[1]}`")
            lines.append(f"- Stop loss: `{r.stop_loss}`")
            lines.append(f"- Take profit hint: `{r.take_profit_hint[0]} - {r.take_profit_hint[1]}`")
            lines.append(f"- Risk flags: {', '.join(r.risk_flags) if r.risk_flags else 'None'}")
            lines.append(f"- Rationale: {', '.join(r.rationale) if r.rationale else 'None'}")
            lines.append("")
    else:
        lines.append("- No candidates passed the score threshold.")
        lines.append("")

    lines.append("## Risk")
    for k, v in plan.risk.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Daily No-Go Rules")
    for item in plan.daily_no_go:
        lines.append(f"- {item}")
    if not plan.daily_no_go:
        lines.append("- None")
    lines.append("")

    lines.append("## Reflection")
    for k, v in plan.reflection.items():
        lines.append(f"### {k}")
        if isinstance(v, list):
            if v:
                for item in v:
                    lines.append(f"- {item}")
            else:
                lines.append("- None")
        else:
            lines.append(f"- {v}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_summary(plan: DailyPlan) -> None:
    print("=" * 72)
    print(f"Stock Agent MVP | date={plan.date} | mode={plan.mode}")
    print("=" * 72)
    print(f"Market regime      : {plan.market_regime}")
    print(f"Trading permission : {plan.trading_permission}")
    print("")

    print("Top sectors:")
    if plan.top_sectors:
        for s in plan.top_sectors:
            print(f"  - {s['name']} (rank={s['rank']}, strength={s['strength_score']}, stage={s['stage']})")
    else:
        print("  - None")
    print("")

    print("Recommendations:")
    if plan.recommendations:
        for r in plan.recommendations:
            print(
                f"  - {r.ticker}: score={r.score}, setup={r.setup}, "
                f"entry={r.entry_zone[0]}-{r.entry_zone[1]}, stop={r.stop_loss}"
            )
    else:
        print("  - No candidates passed the threshold.")
    print("")

    print("Risk:")
    for k, v in plan.risk.items():
        print(f"  - {k}: {v}")
    print("")

    print("No-go rules:")
    if plan.daily_no_go:
        for item in plan.daily_no_go:
            print(f"  - {item}")
    else:
        print("  - None")
    print("=" * 72)


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file stock agent MVP")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use Yahoo Finance live daily data",
    )
    parser.add_argument(
        "--watchlist",
        nargs="*",
        default=DEFAULT_WATCHLIST,
        help="Custom watchlist tickers",
    )
    parser.add_argument(
        "--consecutive-losses",
        type=int,
        default=0,
        help="Used by the risk agent to reduce exposure after repeated losses",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to external strategy config JSON",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run rolling backtest and output summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    global APP_CONFIG, RISK_CONFIG, SECTOR_SCORING_CONFIG, MARKET_REGIME_CONFIG, BACKTEST_CONFIG
    APP_CONFIG = load_app_config(Path(args.config))
    RISK_CONFIG = APP_CONFIG["risk"]
    SECTOR_SCORING_CONFIG = APP_CONFIG["sector_scoring"]
    MARKET_REGIME_CONFIG = APP_CONFIG["market_regime"]
    BACKTEST_CONFIG = APP_CONFIG["backtest"]

    if args.backtest:
        summary = run_backtest(args.watchlist, consecutive_losses=args.consecutive_losses)
        print("=" * 72)
        print("Backtest Summary")
        print("=" * 72)
        for k, v in asdict(summary).items():
            print(f"{k}: {v}")
        return

    mode = "live" if args.live else "mock"

    try:
        plan = build_daily_plan(
            mode=mode,
            watchlist=args.watchlist,
            consecutive_losses=args.consecutive_losses,
        )
    except Exception as e:
        print(f"[ERROR] Failed in mode={mode}: {e}")
        if mode == "live":
            print("[INFO] Falling back to mock mode.")
            plan = build_daily_plan(
                mode="mock",
                watchlist=args.watchlist,
                consecutive_losses=args.consecutive_losses,
            )
        else:
            raise

    json_path = write_json_report(plan)
    md_path = write_markdown_report(plan)
    print_summary(plan)
    print(f"JSON report written to: {json_path}")
    print(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    main()
