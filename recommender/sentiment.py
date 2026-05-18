"""Lightweight sentiment analysis (offline, no ML deps)."""
import re
from dataclasses import dataclass
from typing import Dict, List


POSITIVE = {
    "love", "loved", "amazing", "awesome", "great", "excellent", "fantastic",
    "wonderful", "brilliant", "beautiful", "enjoyed", "enjoy", "best", "good",
    "happy", "fun", "funny", "heartwarming", "masterpiece", "perfect",
}
NEGATIVE = {
    "hate", "hated", "bad", "terrible", "awful", "boring", "worst", "poor",
    "disappointing", "disappointed", "waste", "dull", "annoying", "stupid",
    "horrible", "trash", "slow", "confusing", "overrated",
}


@dataclass(frozen=True)
class SentimentResult:
    label: str  # positive | neutral | negative
    score: float  # -1 .. 1
    confidence: float  # 0 .. 1


def analyze_sentiment(text: str) -> SentimentResult:
    raw = (text or "").strip().lower()
    if not raw:
        return SentimentResult(label="neutral", score=0.0, confidence=0.0)

    words = set(re.findall(r"\b[\w']+\b", raw))
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)

    if pos == 0 and neg == 0:
        return SentimentResult(label="neutral", score=0.0, confidence=0.35)

    total = pos + neg
    score = (pos - neg) / max(1, total)
    confidence = min(1.0, 0.4 + 0.15 * total)

    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(label=label, score=round(score, 3), confidence=round(confidence, 3))


def sentiment_to_dict(result: SentimentResult) -> Dict:
    return {"label": result.label, "score": result.score, "confidence": result.confidence}
