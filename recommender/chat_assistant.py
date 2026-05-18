import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ai_provider import maybe_enhance_reply
from .fake_review import analyze_movie_reviews, analyze_review_text
from .sentiment import analyze_sentiment, sentiment_to_dict


@dataclass
class ChatResponse:
    text: str
    items: List[Dict]
    intent: str
    sentiment: Optional[Dict[str, Any]] = None
    review_analysis: Optional[Dict[str, Any]] = None
    typing_ms: int = 900
    extras: Dict[str, Any] = field(default_factory=dict)


_LIKE_RE = re.compile(
    r"\b(?:movies?\s+like|similar\s+to|like)\s+(?P<title>.+?)(?:\s*\?)?$",
    re.IGNORECASE,
)
_SUGGEST_RE = re.compile(
    r"\b(?:suggest|recommend|show)\s+(?P<genre>[a-zA-Z\- ]+?)\s+movies?\b",
    re.IGNORECASE,
)
_DETAILS_RE = re.compile(
    r"\b(?:about|details?|plot|story(?:line)?|info)\s+(?:for|on|of)?\s*(?P<title>.+?)(?:\s*\?)?$",
    re.IGNORECASE,
)
_STREAM_RE = re.compile(
    r"\b(?:where\s+(?:can\s+)?(?:i\s+)?watch|stream(?:ing)?|platform)\s+(?P<title>.+?)(?:\s*\?)?$",
    re.IGNORECASE,
)
_FAKE_RE = re.compile(
    r"\b(?:fake\s+review|check\s+review|analyze\s+review|is\s+this\s+fake)\b",
    re.IGNORECASE,
)
_FAKE_MOVIE_RE = re.compile(
    r"\bfake\s+reviews?\s+(?:for|on|about)\s+(?P<title>.+?)(?:\s*\?)?$",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings)(?:\s|!|,|$)",
    re.IGNORECASE,
)
_MOOD_RE = re.compile(
    r"\b(?:i\s+am|i'm|feeling|feel)\s+(?P<mood>happy|sad|bored|stressed|scared|romantic|relaxed|excited|angry)\b",
    re.IGNORECASE,
)
_PERSONAL_RE = re.compile(
    r"\b(?:for\s+me|personalized|my\s+taste|based\s+on\s+my\s+history|watch\s+history)\b",
    re.IGNORECASE,
)

MOOD_GENRES: Dict[str, List[str]] = {
    "happy": ["Comedy", "Animation", "Family"],
    "sad": ["Drama", "Romance"],
    "bored": ["Action", "Adventure", "Comedy"],
    "stressed": ["Comedy", "Animation", "Family"],
    "scared": ["Horror", "Thriller"],
    "romantic": ["Romance", "Drama"],
    "relaxed": ["Documentary", "Animation", "Family"],
    "excited": ["Action", "Adventure", "Science Fiction"],
    "angry": ["Action", "Thriller"],
}

STREAMING_PLATFORMS = [
    "Netflix",
    "Amazon Prime Video",
    "Disney+",
    "Hulu",
    "Max",
    "Apple TV+",
]


def _normalize_genre(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _overview_short(text: Any, max_len: int = 180) -> str:
    s = str(text or "").strip()
    if not s or s.lower() == "nan":
        return "No storyline available in our catalog."
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _trailer_url(title: str) -> str:
    q = urllib.parse.quote_plus(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={q}"


def _streaming_for_title(title: str) -> List[str]:
    seed = sum(ord(c) for c in (title or "")) % len(STREAMING_PLATFORMS)
    n = 2 + (seed % 2)
    out = []
    for i in range(n):
        out.append(STREAMING_PLATFORMS[(seed + i) % len(STREAMING_PLATFORMS)])
    return list(dict.fromkeys(out))


def _movie_row_to_item(movie, include_streaming: bool = True) -> Dict[str, Any]:
    title = movie.get("title") if hasattr(movie, "get") else getattr(movie, "title", "Movie")
    genres = movie.get("genres") if hasattr(movie, "get") else getattr(movie, "genres", [])
    labels = _iter_genre_labels(genres)
    genres_str = ", ".join(labels[:4]) if labels else "N/A"

    poster = movie.get("poster_path")
    poster_url = None
    if poster is not None and str(poster) not in ("", "nan"):
        poster_url = f"https://image.tmdb.org/t/p/w342{poster}"

    vote_avg = movie.get("vote_average")
    rating = f"{float(vote_avg):.1f}/10" if vote_avg is not None and str(vote_avg) != "nan" else "N/A"
    votes = movie.get("vote_count")
    votes_str = f"{int(votes):,}" if votes is not None and str(votes) != "nan" else "N/A"

    item = {
        "title": title,
        "release_date": str(movie.get("release_date", "Unknown"))
        if str(movie.get("release_date", "")) not in ("", "nan")
        else "Unknown",
        "production": str(movie.get("primary_company", "Unknown"))
        if str(movie.get("primary_company", "")) not in ("", "nan")
        else "Unknown",
        "genres": genres_str,
        "rating": rating,
        "votes": votes_str,
        "overview": _overview_short(movie.get("overview")),
        "poster_url": poster_url,
        "trailer_url": _trailer_url(str(title)),
        "google_link": f"https://www.google.com/search?q={urllib.parse.quote_plus(str(title) + ' movie')}",
        "imdb_link": (
            f"https://www.imdb.com/title/{movie.get('imdb_id')}"
            if movie.get("imdb_id") and str(movie.get("imdb_id")) not in ("", "nan")
            else None
        ),
    }
    if include_streaming:
        item["streaming"] = _streaming_for_title(str(title))
    return item


def _recommendation_dict_to_item(d: Dict) -> Dict[str, Any]:
    title = d.get("title", "Movie")
    return {
        "title": title,
        "release_date": d.get("release_date", "Unknown"),
        "production": d.get("production", "Unknown"),
        "genres": d.get("genres", "N/A"),
        "rating": d.get("rating", "N/A"),
        "votes": d.get("votes", "N/A"),
        "overview": d.get("overview", "Similar pick from our catalog."),
        "poster_url": d.get("poster_url"),
        "trailer_url": _trailer_url(title),
        "google_link": d.get("google_link"),
        "imdb_link": d.get("imdb_link"),
        "streaming": _streaming_for_title(title),
        "similarity_score": d.get("similarity_score"),
    }


def _find_movie_row(recommender, title: str):
    matched = recommender.find_movie(title)
    if not matched:
        return None, None
    idx = recommender.title_to_idx[matched]
    return matched, recommender.metadata.iloc[idx]


def _iter_genre_labels(genres_cell) -> List[str]:
    if genres_cell is None:
        return []
    if hasattr(genres_cell, "__iter__") and not isinstance(genres_cell, str):
        out = []
        for x in genres_cell:
            if x is not None and str(x).strip() and str(x).lower() != "nan":
                out.append(str(x).strip())
        return out
    if isinstance(genres_cell, str) and genres_cell.strip():
        return [x.strip() for x in re.split(r"[|,;/]", genres_cell) if x.strip()]
    return []


def _available_genres_from_metadata(metadata) -> List[str]:
    genres = set()
    if metadata is None or "genres" not in metadata.columns:
        return []
    for g in metadata["genres"].dropna().tolist():
        for x in _iter_genre_labels(g):
            genres.add(x.title() if x.islower() else x)
    return sorted(genres)


def _match_genre(requested: str, available: List[str]) -> Optional[str]:
    req = _normalize_genre(requested)
    if not req:
        return None
    normalized = {_normalize_genre(a): a for a in available}
    if req in normalized:
        return normalized[req]
    for key, original in normalized.items():
        if req in key or key in req:
            return original
    return None


def _genre_recommendations(recommender, genre_name: str, n: int = 8) -> Tuple[str, List[Dict]]:
    available = _available_genres_from_metadata(recommender.metadata)
    matched = _match_genre(genre_name, available)
    if not matched:
        sample = ", ".join(available[:12]) + ("…" if len(available) > 12 else "")
        return f"I couldn't find genre “{genre_name}”. Try: {sample}", []

    md = recommender.metadata
    matched_lower = matched.lower()

    def _has_genre(g):
        return matched_lower in [x.lower() for x in _iter_genre_labels(g)]

    mask = md["genres"].apply(_has_genre)
    subset = md[mask].copy()
    if subset.empty:
        return f"No “{matched}” movies found in the catalog.", []

    for col in ("vote_average", "vote_count"):
        if col in subset.columns:
            subset[col] = subset[col].fillna(0)
    subset = subset.sort_values(
        [c for c in ["vote_average", "vote_count"] if c in subset.columns],
        ascending=False,
    )
    items = [_movie_row_to_item(row) for _, row in subset.head(n).iterrows()]
    return f"Top {len(items)} **{matched}** picks for you:", items


def _personalized_recommendations(recommender, context: Dict, n: int = 8) -> Tuple[str, List[Dict]]:
    watched = context.get("watched_titles") or []
    search_hist = context.get("search_history") or []
    seeds = list(dict.fromkeys(list(watched) + list(search_hist)))[:5]

    if not seeds:
        return (
            "Mark movies as watched or search for titles first — then I can personalize picks for you.",
            [],
        )

    genre_counts: Dict[str, int] = {}
    for title in seeds:
        _, row = _find_movie_row(recommender, title)
        if row is None:
            continue
        gs = row.get("genres")
        for g in _iter_genre_labels(gs)[:2]:
            key = g.title() if g.islower() else g
            genre_counts[key] = genre_counts.get(key, 0) + 1

    if not genre_counts:
        result = recommender.get_recommendations(seeds[0], n=n)
        if "error" in result:
            return result["error"], []
        items = [_recommendation_dict_to_item(x) for x in result.get("recommendations", [])]
        return f"Based on your interest in “{seeds[0]}”, you might like:", items

    top_genre = sorted(genre_counts.items(), key=lambda x: -x[1])[0][0]
    text, items = _genre_recommendations(recommender, top_genre, n=n)
    return f"Personalized for you (genres from your history: {top_genre}): " + text.replace("**", ""), items


def _extract_intent(message: str) -> Tuple[str, Dict[str, str]]:
    msg = (message or "").strip()
    if not msg:
        return "empty", {}

    if _GREETING_RE.search(msg):
        return "greeting", {}

    m = _MOOD_RE.search(msg)
    if m:
        return "mood", {"mood": m.group("mood").lower()}

    m = _FAKE_MOVIE_RE.search(msg)
    if m:
        return "fake_reviews_movie", {"title": m.group("title").strip().strip("\"'")}

    if _FAKE_RE.search(msg):
        return "fake_review_text", {"text": msg}

    if _PERSONAL_RE.search(msg):
        return "personalized", {}

    m = _STREAM_RE.search(msg)
    if m:
        return "streaming", {"title": m.group("title").strip().strip("\"'")}

    m = _DETAILS_RE.search(msg)
    if m:
        return "details", {"title": m.group("title").strip().strip("\"'")}

    m = _LIKE_RE.search(msg)
    if m:
        title = (m.group("title") or "").strip().strip("\"'")
        if title:
            return "similar", {"title": title}

    m = _SUGGEST_RE.search(msg)
    if m:
        genre = _normalize_genre(m.group("genre"))
        if genre:
            return "genre", {"genre": genre}

    return "search", {"query": msg}


def handle_message(
    recommender,
    message: str,
    n: int = 8,
    session_context: Optional[Dict[str, Any]] = None,
    use_ai: bool = True,
) -> ChatResponse:
    context = session_context or {}
    history = context.get("chat_history") or []
    intent, params = _extract_intent(message)
    user_sentiment = sentiment_to_dict(analyze_sentiment(message))

    if intent == "empty":
        return ChatResponse(
            text="Ask me: “Suggest thriller movies”, “Movies like Inception”, “I'm feeling sad”, or “Check fake reviews for Avatar”.",
            items=[],
            intent=intent,
            sentiment=user_sentiment,
        )

    if recommender is None:
        return ChatResponse(
            text="The recommendation model is still loading. Please try again in a moment.",
            items=[],
            intent="loading",
            sentiment=user_sentiment,
            typing_ms=600,
        )

    resp: ChatResponse

    if intent == "greeting":
        resp = ChatResponse(
            text=(
                "Hello! I'm your AI Movie Assistant. I can suggest by genre or mood, "
                "find similar films, check fake reviews, show posters & trailers, and more. What are you in the mood for?"
            ),
            items=[],
            intent=intent,
            sentiment=user_sentiment,
            typing_ms=700,
        )

    elif intent == "mood":
        mood = params.get("mood", "")
        genres = MOOD_GENRES.get(mood, ["Drama", "Comedy"])
        all_items: List[Dict] = []
        for g in genres[:2]:
            _, items = _genre_recommendations(recommender, g, n=max(4, n // 2))
            all_items.extend(items)
        seen = set()
        unique = []
        for it in all_items:
            if it["title"] not in seen:
                seen.add(it["title"])
                unique.append(it)
        resp = ChatResponse(
            text=f"Since you're feeling **{mood}**, here are some picks that might match your mood:",
            items=unique[:n],
            intent=intent,
            sentiment=user_sentiment,
        )

    elif intent == "fake_reviews_movie":
        title = params["title"]
        matched, _ = _find_movie_row(recommender, title)
        query = matched or title
        analysis = analyze_movie_reviews(query)
        warn = ""
        if analysis.get("has_fake_warning"):
            warn = f" ⚠ {analysis['fake_pct']}% of sampled reviews look suspicious."
        resp = ChatResponse(
            text=f"Fake review scan for “{query}”: {analysis['trusted_count']} trusted, {analysis['fake_count']} flagged.{warn}",
            items=[],
            intent=intent,
            sentiment=user_sentiment,
            review_analysis=analysis,
            typing_ms=1200,
        )

    elif intent == "fake_review_text":
        review_text = params.get("text", message)
        m = re.search(r'["""](.+?)["""]', review_text)
        if m:
            review_text = m.group(1)
        detection = analyze_review_text(review_text)
        label = detection.label
        pct = round(detection.score * 100)
        reasons = ", ".join(detection.reasons) if detection.reasons else "none"
        rev_sent = sentiment_to_dict(analyze_sentiment(review_text))
        resp = ChatResponse(
            text=f"Review looks **{label}** ({pct}% fake suspicion). Signals: {reasons}.",
            items=[],
            intent=intent,
            sentiment=user_sentiment,
            review_analysis={
                "label": label,
                "score": detection.score,
                "score_pct": pct,
                "reasons": detection.reasons,
                "review_sentiment": rev_sent,
            },
            typing_ms=1000,
        )

    elif intent == "personalized":
        text, items = _personalized_recommendations(recommender, context, n=n)
        resp = ChatResponse(text=text, items=items, intent=intent, sentiment=user_sentiment)

    elif intent == "streaming":
        title = params["title"]
        matched, row = _find_movie_row(recommender, title)
        if row is None:
            resp = ChatResponse(
                text=f"I couldn't find “{title}” in our catalog.",
                items=[],
                intent="streaming_none",
                sentiment=user_sentiment,
            )
        else:
            item = _movie_row_to_item(row)
            platforms = ", ".join(item.get("streaming", []))
            resp = ChatResponse(
                text=f"Where to watch “{matched}” (availability may vary by region): {platforms}",
                items=[item],
                intent=intent,
                sentiment=user_sentiment,
            )

    elif intent == "details":
        title = params["title"]
        matched, row = _find_movie_row(recommender, title)
        if row is None:
            resp = ChatResponse(
                text=f"I couldn't find details for “{title}”.",
                items=[],
                intent="details_none",
                sentiment=user_sentiment,
            )
        else:
            item = _movie_row_to_item(row)
            resp = ChatResponse(
                text=(
                    f"**{matched}** — {item['rating']} ({item['votes']} votes) · {item['genres']}\n\n"
                    f"{item['overview']}"
                ),
                items=[item],
                intent=intent,
                sentiment=user_sentiment,
            )

    elif intent == "similar":
        title = params["title"]
        result = recommender.get_recommendations(title, n=n)
        if "error" in result:
            sugg = result.get("suggestions") or []
            extra = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
            resp = ChatResponse(text=result["error"] + extra, items=[], intent=intent, sentiment=user_sentiment)
        else:
            items = [_recommendation_dict_to_item(x) for x in result.get("recommendations", [])]
            resp = ChatResponse(
                text=f"Movies similar to “{result['query_movie']}”:",
                items=items,
                intent=intent,
                sentiment=user_sentiment,
            )

    elif intent == "genre":
        text, items = _genre_recommendations(recommender, params["genre"], n=n)
        resp = ChatResponse(text=text.replace("**", ""), items=items, intent=intent, sentiment=user_sentiment)

    else:
        query = params.get("query") or message
        matches = recommender.search_movies(query, n=min(12, max(5, n)))
        if not matches:
            resp = ChatResponse(
                text=f"No matches for “{query}”. Try “Movies like Inception” or “Suggest action movies”.",
                items=[],
                intent="search_none",
                sentiment=user_sentiment,
            )
        else:
            items = []
            for m_title in matches[:n]:
                _, row = _find_movie_row(recommender, m_title)
                if row is not None:
                    items.append(_movie_row_to_item(row))
                else:
                    items.append({"title": m_title, "trailer_url": _trailer_url(m_title)})
            resp = ChatResponse(
                text="Here are matching titles — ask “Movies like <title>” or “Details about <title>”.",
                items=items,
                intent="search",
                sentiment=user_sentiment,
            )

    if use_ai and intent not in ("fake_review_text", "fake_reviews_movie", "loading"):
        resp.text = maybe_enhance_reply(message, history, resp.text.replace("**", ""), resp.intent)

    resp.text = resp.text.replace("**", "")
    resp.sentiment = user_sentiment
    return resp
