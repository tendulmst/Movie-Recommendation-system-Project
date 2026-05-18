import hashlib
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


PROMO_WORDS = {
    "buy",
    "discount",
    "offer",
    "free",
    "subscribe",
    "promo",
    "promotion",
    "click",
    "link",
    "whatsapp",
    "telegram",
    "dm",
    "inbox",
    "guaranteed",
}

GENERIC_PHRASES = {
    "best movie ever",
    "must watch",
    "highly recommended",
    "worth watching",
    "amazing",
    "awesome",
    "excellent",
    "superb",
    "mind blowing",
}


@dataclass(frozen=True)
class FakeReviewResult:
    score: float  # 0..1 (higher = more likely fake)
    label: str  # "fake" | "real"
    reasons: List[str]
    features: Dict[str, Any]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def analyze_review_text(text: str) -> FakeReviewResult:
    """
    Lightweight heuristic fake-review detection.
    No external dependencies; works offline.
    """
    raw = (text or "").strip()
    t = raw.lower()

    # Basic stats
    length = len(raw)
    word_count = len(re.findall(r"\b[\w']+\b", t))
    exclam = raw.count("!")
    qmarks = raw.count("?")
    urls = len(re.findall(r"https?://|www\.", t))
    emojis = len(re.findall(r"[\U0001F300-\U0001FAFF]", raw))
    repeated_chars = len(re.findall(r"(.)\1{3,}", raw))
    repeated_words = len(re.findall(r"\b(\w+)\b(?:\s+\1\b){2,}", t))
    all_caps_ratio = 0.0
    letters = re.findall(r"[A-Za-z]", raw)
    if letters:
        caps = sum(1 for c in letters if c.isupper())
        all_caps_ratio = caps / max(1, len(letters))

    promo_hits = sum(1 for w in PROMO_WORDS if re.search(rf"\b{re.escape(w)}\b", t))
    generic_hits = sum(1 for p in GENERIC_PHRASES if p in t)

    # Scoring: each feature adds suspicion; keep it interpretable.
    score = 0.0
    reasons: List[str] = []

    if word_count <= 3:
        score += 0.35
        reasons.append("Very short review")
    elif word_count <= 8:
        score += 0.18
        reasons.append("Short / low-detail review")

    if exclam >= 4:
        score += 0.18
        reasons.append("Excessive exclamation marks")
    elif exclam >= 2:
        score += 0.08

    if urls > 0:
        score += 0.30
        reasons.append("Contains a link")

    if promo_hits > 0:
        score += 0.30
        reasons.append("Promotional / spammy wording")

    if generic_hits > 0:
        score += min(0.22, 0.11 * generic_hits)
        reasons.append("Generic praise (low specificity)")

    if all_caps_ratio >= 0.6 and length >= 12:
        score += 0.18
        reasons.append("Mostly ALL CAPS")

    if repeated_chars > 0:
        score += 0.12
        reasons.append("Repeated characters (e.g., 'soooo')")

    if repeated_words > 0:
        score += 0.12
        reasons.append("Repeated words / unnatural repetition")

    if emojis >= 4:
        score += 0.10
        reasons.append("Excessive emojis")

    if qmarks >= 4:
        score += 0.06

    score = _clamp01(score)
    label = "fake" if score >= 0.65 else "real"

    if not raw:
        return FakeReviewResult(
            score=0.0,
            label="real",
            reasons=["Empty review"],
            features={
                "length": 0,
                "word_count": 0,
            },
        )

    return FakeReviewResult(
        score=score,
        label=label,
        reasons=reasons[:6],
        features={
            "length": length,
            "word_count": word_count,
            "exclamations": exclam,
            "question_marks": qmarks,
            "links": urls,
            "promo_hits": promo_hits,
            "generic_hits": generic_hits,
            "all_caps_ratio": round(all_caps_ratio, 3),
            "repeated_chars": repeated_chars,
            "repeated_words": repeated_words,
            "emoji_count": emojis,
        },
    )


_DEMO_AUTHORS = (
    "Alex M.",
    "Jordan K.",
    "Sam R.",
    "Taylor P.",
    "Casey L.",
    "Morgan D.",
    "Riley N.",
    "Jamie W.",
)

_DEMO_REVIEWS_TRUSTED = (
    "The second act slows down, but strong performances and thoughtful direction kept me engaged through the end.",
    "Not perfect — a few plot holes — yet the score and visuals make it a solid weekend watch.",
    "I liked how the film balances humor with quieter character moments; felt genuine rather than formulaic.",
    "Cinematography stands out, especially in the night scenes. Pacing could be tighter in the middle.",
    "Went in skeptical and left impressed. The lead chemistry carries scenes that would otherwise feel thin.",
    "A mature take on the genre. The ending is bittersweet but earned, not just shock value.",
)

_DEMO_REVIEWS_SUSPICIOUS = (
    "BEST MOVIE EVER!!! MUST WATCH!!! Click www.free-tickets.com for AMAZING deals!!!",
    "Amazing awesome excellent superb mind blowing!!! Subscribe now for more reviews!!!",
    "So good!!!",
    "HIGHLY RECOMMENDED!!! Best movie ever!!! DM me on telegram for full list!!!",
    "Worth watching!!! Guaranteed 5 stars!!! Promo code inside — click the link!!!",
    "Mind blowing amazing excellent!!! Free download at www.example.com!!!",
)


def _seed_for_title(title: str) -> int:
    return int(hashlib.md5((title or "").strip().lower().encode("utf-8")).hexdigest()[:8], 16)


def get_demo_reviews_for_movie(movie_title: str, count: int = 8) -> List[Dict[str, str]]:
    """Deterministic sample reviews for a movie (demo / offline dataset)."""
    seed = _seed_for_title(movie_title)
    trusted = list(_DEMO_REVIEWS_TRUSTED)
    suspicious = list(_DEMO_REVIEWS_SUSPICIOUS)
    n_trusted = 4 + (seed % 3)
    n_suspicious = max(2, count - n_trusted)
    n_trusted = min(n_trusted, count - 2)

    picked: List[Dict[str, str]] = []
    for i in range(n_trusted):
        text = trusted[(seed + i * 3) % len(trusted)]
        author = _DEMO_AUTHORS[i % len(_DEMO_AUTHORS)]
        picked.append({"author": author, "text": text, "source": "community"})

    for j in range(n_suspicious):
        text = suspicious[(seed + j * 5 + 1) % len(suspicious)]
        author = _DEMO_AUTHORS[(n_trusted + j) % len(_DEMO_AUTHORS)]
        picked.append({"author": author, "text": text, "source": "community"})

    return picked[:count]


def analyze_movie_reviews(
    movie_title: str,
    extra_reviews: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Analyze all reviews for a movie; split into trusted vs likely fake."""
    raw_items = get_demo_reviews_for_movie(movie_title)
    if extra_reviews:
        for item in extra_reviews:
            if item and (item.get("text") or "").strip():
                raw_items.append(
                    {
                        "author": item.get("author") or "You",
                        "text": (item.get("text") or "").strip(),
                        "source": item.get("source") or "session",
                    }
                )

    analyzed: List[Dict[str, Any]] = []
    for item in raw_items:
        result = analyze_review_text(item["text"])
        analyzed.append(
            {
                "author": item.get("author") or "Anonymous",
                "text": item["text"],
                "source": item.get("source") or "community",
                "label": result.label,
                "score": round(result.score, 3),
                "score_pct": round(result.score * 100),
                "reasons": result.reasons,
            }
        )

    trusted = [r for r in analyzed if r["label"] == "real"]
    fake = [r for r in analyzed if r["label"] == "fake"]
    total = len(analyzed)

    return {
        "movie_title": movie_title,
        "total": total,
        "trusted_count": len(trusted),
        "fake_count": len(fake),
        "trusted_reviews": trusted,
        "fake_reviews": fake,
        "has_fake_warning": len(fake) > 0,
        "fake_pct": round(100 * len(fake) / total) if total else 0,
        "trusted_pct": round(100 * len(trusted) / total) if total else 0,
    }

