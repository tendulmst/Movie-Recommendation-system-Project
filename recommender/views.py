"""
Movie Recommendation System Views
Integrates with advanced TMDB model training system
"""
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional
from difflib import get_close_matches

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import json
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .fake_review import analyze_review_text, analyze_movie_reviews
from .chat_assistant import handle_message
from .sentiment import analyze_sentiment, sentiment_to_dict

try:
    from .models import ChatLog, ReviewCheckLog
except Exception:
    ChatLog = None
    ReviewCheckLog = None

 

logger = logging.getLogger(__name__)


def _get_ai_provider() -> str:
    return (os.environ.get("CHAT_AI_PROVIDER") or getattr(settings, "CHAT_AI_PROVIDER", "") or "").strip().lower()


# Global cache for recommender system
_RECOMMENDER = None
_MODEL_LOADING = False
_MODEL_LOAD_PROGRESS = 0
_LOADING_THREAD = None
_LOAD_ERROR = None


class MovieRecommender:
    """Integrated recommender system matching training/infer.py logic"""
    
    def __init__(self, model_dir='models', progress_callback=None):
        """Initialize with trained model directory"""
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.similarity_matrix = None
        self.title_to_idx = None
        self.config = None
        self._load_models(progress_callback)
    
    def _load_models(self, progress_callback=None):
        """Load all model artifacts with progress tracking"""
        global _MODEL_LOAD_PROGRESS
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load metadata (25%)
        if progress_callback:
            progress_callback(10)
        self.metadata = pd.read_parquet(self.model_dir / 'movie_metadata.parquet')
        if progress_callback:
            progress_callback(25)
        
        # Load similarity matrix (sparse or dense) (50%)
        if progress_callback:
            progress_callback(40)
        if (self.model_dir / 'similarity_matrix.npz').exists():
            self.similarity_matrix = load_npz(self.model_dir / 'similarity_matrix.npz').toarray()
        else:
            self.similarity_matrix = np.load(self.model_dir / 'similarity_matrix.npy')
        if progress_callback:
            progress_callback(65)
        
        # Load title mapping (75%)
        with open(self.model_dir / 'title_to_idx.json', 'r') as f:
            self.title_to_idx = json.load(f)
        if progress_callback:
            progress_callback(80)
        
        # Load config (100%)
        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Loaded {self.config['n_movies']:,} movies successfully")
    
    def find_movie(self, title: str) -> Optional[str]:
        """Find closest matching movie title"""
        matches = get_close_matches(title, self.title_to_idx.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None

    
    def search_movies(self, query: str, n: int = 20) -> List[str]:
        """Search movies by partial title"""
        query_lower = query.lower()
        return [title for title in self.title_to_idx.keys() 
                if query_lower in title.lower()][:n]
    
    def get_recommendations(
        self,
        movie_title: str,
        n: int = 15,
        min_rating: float = None
    ) -> Dict:
        """Get movie recommendations with optional filtering"""
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            return {'error': f"Movie '{movie_title}' not found", 'suggestions': self.search_movies(movie_title, 5)}
        
        movie_idx = self.title_to_idx[matched_title]
        source_movie = self.metadata.iloc[movie_idx]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude self
        
        recommendations = []
        for idx, score in sim_scores:
            if len(recommendations) >= n:
                break
            
            movie = self.metadata.iloc[idx]
            
            # Rating filter
            if min_rating and movie['vote_average'] < min_rating:
                continue
            
            recommendations.append({
                'title': movie['title'],
                'release_date': movie['release_date'] if pd.notna(movie['release_date']) else 'Unknown',
                'production': movie['primary_company'] if pd.notna(movie['primary_company']) else 'Unknown',
                'genres': ', '.join(movie['genres'][:3]) if isinstance(movie['genres'], list) else 'N/A',
                'rating': f"{movie['vote_average']:.1f}/10" if pd.notna(movie['vote_average']) else 'N/A',
                'votes': f"{movie['vote_count']:,}" if pd.notna(movie['vote_count']) else 'N/A',
                'similarity_score': f"{score:.3f}",
                'imdb_id': movie['imdb_id'] if pd.notna(movie['imdb_id']) else None,
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if pd.notna(movie['poster_path']) else None,
                'google_link': f"https://www.google.com/search?q={'+'.join(movie['title'].split())}+movie",
                'imdb_link': f"https://www.imdb.com/title/{movie['imdb_id']}" if pd.notna(movie['imdb_id']) else None
            })
        
        return {
            'query_movie': matched_title,
            'source_movie': {
                'production': source_movie['primary_company'] if pd.notna(source_movie['primary_company']) else 'Unknown',
                'rating': f"{source_movie['vote_average']:.1f}/10" if pd.notna(source_movie['vote_average']) else 'N/A',
                'genres': ', '.join(source_movie['genres'][:3]) if isinstance(source_movie['genres'], list) else 'N/A'
            },
            'recommendations': recommendations
        }


def _load_model_in_background():
    """Load model in background thread"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    _MODEL_LOADING = True
    _MODEL_LOAD_PROGRESS = 0
    _LOAD_ERROR = None
    
    # Check for model directory (configurable via settings or environment)
    model_dir =  'models'
    
    # Fallback to static directory if models directory doesn't exist
    if not Path(model_dir).exists():
        model_dir = 'static'
        logger.warning(f"Model directory not found, using static directory")
    
    try:
        def progress_callback(progress):
            global _MODEL_LOAD_PROGRESS
            _MODEL_LOAD_PROGRESS = progress
            logger.info(f"Model loading progress: {progress}%")
        
        _RECOMMENDER = MovieRecommender(model_dir, progress_callback)
        _MODEL_LOADING = False
        _MODEL_LOAD_PROGRESS = 100
        logger.info("Model loaded successfully")
    except Exception as e:
        _MODEL_LOADING = False
        _LOAD_ERROR = str(e)
        logger.error(f"Failed to load recommender: {e}")


def _start_model_loading():
    """Start model loading in background if not already started"""
    global _LOADING_THREAD, _RECOMMENDER, _MODEL_LOADING
    
    if _RECOMMENDER is None and not _MODEL_LOADING:
        if _LOADING_THREAD is None or not _LOADING_THREAD.is_alive():
            logger.info("Starting model loading in background...")
            _LOADING_THREAD = threading.Thread(target=_load_model_in_background, daemon=True)
            _LOADING_THREAD.start()


def _get_recommender():
    """Get or initialize the recommender singleton"""
    global _RECOMMENDER, _LOAD_ERROR
    
    if _RECOMMENDER is None:
        _start_model_loading()
        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)
        return None
    
    return _RECOMMENDER


@require_http_methods(["GET", "POST"])
def main(request):
    """
    Main view for movie recommendation system.
    GET: Display search interface
    POST: Process search and display recommendations
    """
    # Start loading model if not already loading/loaded
    _start_model_loading()
    
    recommender = _get_recommender()
    
    # If model is still loading, show the page with loading state
    if recommender is None:
        if request.method == 'GET':
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
            })
        else:
            # For POST requests, return error if model not ready
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
                'error_message': 'Model is still loading. Please wait a moment and try again.',
            })
    
    # Model is loaded, proceed normally
    titles_list = list(recommender.title_to_idx.keys())
    
    if request.method == 'GET':
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
            }
        )
    
    # POST request - process search
    movie_name = request.POST.get('movie_name', '').strip()
    
    if not movie_name:
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'error_message': 'Please enter a movie name.',
            }
        )
    
    # Get recommendations
    result = recommender.get_recommendations(movie_name, n=15)
    
    if 'error' in result:
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'input_movie_name': movie_name,
                'error_message': result['error'],
                'suggestions': result.get('suggestions', [])
            }
        )
    
    query_movie = result['query_movie']
    search_hist = request.session.get("search_history", [])
    if not isinstance(search_hist, list):
        search_hist = []
    if query_movie not in search_hist:
        search_hist.append(query_movie)
    request.session["search_history"] = search_hist[-30:]
    request.session.modified = True

    session_reviews = request.session.get("reviews", {})
    extra = []
    if isinstance(session_reviews, dict):
        user_entry = session_reviews.get(query_movie)
        if isinstance(user_entry, dict) and (user_entry.get("review") or "").strip():
            extra.append(
                {
                    "author": "You",
                    "text": user_entry["review"],
                    "source": "your review",
                }
            )

    review_analysis = analyze_movie_reviews(query_movie, extra_reviews=extra or None)

    return render(
        request,
        'recommender/result.html',
        {
            'all_movie_names': titles_list,
            'input_movie_name': query_movie,
            'source_movie': result['source_movie'],
            'recommended_movies': result['recommendations'],
            'total_recommendations': len(result['recommendations']),
            'review_analysis': review_analysis,
        }
    )


@require_http_methods(["GET"])
def search_movies(request):
    """API endpoint for searching movies (autocomplete)"""
    query = request.GET.get('q', '').strip()
    
    if len(query) < 2:
        return JsonResponse({'movies': [], 'count': 0})
    
    try:
        recommender = _get_recommender()
        
        if recommender is None:
            return JsonResponse({'movies': [], 'count': 0, 'loading': True})
        
        matching_movies = recommender.search_movies(query, n=20)
        
        return JsonResponse({
            'movies': matching_movies,
            'count': len(matching_movies)
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return JsonResponse({'error': 'Search failed'}, status=500)


@require_http_methods(["GET"])
def model_status(request):
    """API endpoint to check model loading status"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    # Start loading if not already started
    _start_model_loading()
    
    if _LOAD_ERROR:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'error',
            'error': _LOAD_ERROR
        })
    elif _RECOMMENDER is not None:
        return JsonResponse({
            'loaded': True,
            'progress': 100,
            'status': 'ready'
        })
    elif _MODEL_LOADING:
        return JsonResponse({
            'loaded': False,
            'progress': _MODEL_LOAD_PROGRESS,
            'status': 'loading'
        })
    else:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'initializing'
        })


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for monitoring"""
    try:
        recommender = _get_recommender()
        return JsonResponse({
            'status': 'healthy',
            'movies_loaded': recommender.config['n_movies'],
            'model_dir': str(recommender.model_dir),
            'model_loaded': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)


@require_http_methods(["GET"])
def fake_review_page(request):
    """Redirect: fake review detection runs automatically after movie search."""
    return redirect("recommender:main")


@require_http_methods(["GET"])
def chat_assistant_page(request):
    """Dedicated AI Chatbot Assistant page (separate from home search)."""
    _start_model_loading()
    return render(request, "recommender/assistant.html")


def _json_body(request):
    try:
        return json.loads((request.body or b"{}").decode("utf-8"))
    except Exception:
        return None


@require_http_methods(["POST"])
def fake_review_api(request):
    """API: analyze one review or all reviews for a movie title."""
    payload = _json_body(request)
    if payload is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    movie_title = (payload.get("movie_title") or payload.get("title") or "").strip()
    if movie_title:
        session_reviews = request.session.get("reviews", {})
        extra = []
        if isinstance(session_reviews, dict):
            user_entry = session_reviews.get(movie_title)
            if isinstance(user_entry, dict) and (user_entry.get("review") or "").strip():
                extra.append(
                    {
                        "author": "You",
                        "text": user_entry["review"],
                        "source": "your review",
                    }
                )
        analysis = analyze_movie_reviews(movie_title, extra_reviews=extra or None)
        return JsonResponse(analysis)

    text = (payload.get("text") or "").strip()
    result = analyze_review_text(text)
    return JsonResponse(
        {
            "label": result.label,
            "score": result.score,
            "reasons": result.reasons,
            "features": result.features,
        }
    )


@require_http_methods(["POST"])
def watched_api(request):
    """API: store watched toggle in session (no DB required)."""
    payload = _json_body(request)
    if payload is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = (payload.get("title") or "").strip()
    watched = payload.get("watched", None)
    if not title or not isinstance(watched, bool):
        return JsonResponse({"error": "Expected: { title: string, watched: boolean }"}, status=400)

    watched_set = set(request.session.get("watched_titles", []))
    if watched:
        watched_set.add(title)
    else:
        watched_set.discard(title)
    request.session["watched_titles"] = sorted(watched_set)
    request.session.modified = True

    return JsonResponse({"ok": True, "title": title, "watched": watched})


@require_http_methods(["POST"])
def review_api(request):
    """
    API: accept rating/review and run fake review detection.
    Stored in session for demo purposes (project currently has no DB models).
    """
    payload = _json_body(request)
    if payload is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = (payload.get("title") or "").strip()
    rating = (payload.get("rating") or "").strip()
    review = (payload.get("review") or "").strip()

    if not title:
        return JsonResponse({"error": "Missing title"}, status=400)

    # Rating is optional, but if provided must be 1..5
    if rating:
        try:
            rating_int = int(rating)
        except Exception:
            return JsonResponse({"error": "Rating must be an integer 1..5"}, status=400)
        if rating_int < 1 or rating_int > 5:
            return JsonResponse({"error": "Rating must be between 1 and 5"}, status=400)
    else:
        rating_int = None

    detection = analyze_review_text(review)

    reviews = request.session.get("reviews", {})
    if not isinstance(reviews, dict):
        reviews = {}

    reviews[title] = {
        "rating": rating_int,
        "review": review,
        "fake_review": {
            "label": detection.label,
            "score": detection.score,
            "reasons": detection.reasons,
        },
    }
    request.session["reviews"] = reviews
    request.session.modified = True

    return JsonResponse(
        {
            "ok": True,
            "title": title,
            "saved": True,
            "fake_review": {
                "label": detection.label,
                "score": detection.score,
                "reasons": detection.reasons,
            },
        }
    )


def _session_chat_context(request) -> Dict:
    history = request.session.get("chat_history", [])
    if not isinstance(history, list):
        history = []
    return {
        "chat_history": history,
        "watched_titles": request.session.get("watched_titles", []) or [],
        "search_history": request.session.get("search_history", []) or [],
        "reviews": request.session.get("reviews", {}) or {},
    }


def _append_chat_history(request, user_msg: str, bot_msg: str, intent: str = ""):
    history = request.session.get("chat_history", [])
    if not isinstance(history, list):
        history = []
    history.append({"role": "user", "content": user_msg, "intent": ""})
    history.append({"role": "bot", "content": bot_msg, "intent": intent})
    request.session["chat_history"] = history[-40:]
    request.session.modified = True


def _log_chat(request, role: str, message: str, intent: str = "", sentiment: str = ""):
    if ChatLog is None:
        return
    try:
        ChatLog.objects.create(
            session_key=request.session.session_key or "anonymous",
            role=role,
            message=message[:4000],
            intent=intent or "",
            sentiment=sentiment or "",
        )
    except Exception as e:
        logger.debug("ChatLog save skipped: %s", e)


def _log_review_check(title: str, snippet: str, label: str, score: float, sentiment: str = ""):
    if ReviewCheckLog is None:
        return
    try:
        ReviewCheckLog.objects.create(
            movie_title=title[:255],
            review_snippet=snippet[:2000],
            label=label,
            score=score,
            sentiment=sentiment,
        )
    except Exception as e:
        logger.debug("ReviewCheckLog save skipped: %s", e)


@require_http_methods(["GET"])
def chat_history_api(request):
    """Return stored chat messages for conversational memory."""
    history = request.session.get("chat_history", [])
    if not isinstance(history, list):
        history = []
    return JsonResponse(
        {
            "ok": True,
            "history": history,
            "ai_provider": _get_ai_provider(),
            "ai_enabled": bool(_get_ai_provider()),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def chat_clear_api(request):
    """Clear chat history in session."""
    request.session["chat_history"] = []
    request.session.modified = True
    return JsonResponse(
        {
            "ok": True,
            "history": [],
            "ai_provider": _get_ai_provider(),
            "ai_enabled": bool(_get_ai_provider()),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def chat_assistant_api(request):
    """API: real-time chat — recommendations, mood, fake reviews, sentiment, memory."""
    payload = _json_body(request)
    if payload is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = (payload.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "Missing message"}, status=400)

    if not request.session.session_key:
        request.session.save()

    user_sentiment = sentiment_to_dict(analyze_sentiment(message))
    _log_chat(request, "user", message, sentiment=user_sentiment.get("label", ""))

    _start_model_loading()
    recommender = _get_recommender()

    use_ai = payload.get("use_ai", True)
    resp = handle_message(
        recommender,
        message,
        n=int(payload.get("n") or 8),
        session_context=_session_chat_context(request),
        use_ai=bool(use_ai),
    )

    _append_chat_history(request, message, resp.text, resp.intent)
    _log_chat(
        request,
        "bot",
        resp.text,
        intent=resp.intent,
        sentiment=(resp.sentiment or {}).get("label", ""),
    )

    if resp.review_analysis:
        ra = resp.review_analysis
        if ra.get("label") and ra.get("score") is not None:
            _log_review_check(
                ra.get("movie_title", ""),
                message[:500],
                ra.get("label", ""),
                float(ra.get("score") or 0),
                user_sentiment.get("label", ""),
            )
        elif ra.get("movie_title"):
            _log_review_check(
                ra.get("movie_title", ""),
                "batch scan",
                "fake" if ra.get("has_fake_warning") else "real",
                float(ra.get("fake_pct") or 0) / 100.0,
                user_sentiment.get("label", ""),
            )

    provider = _get_ai_provider()
    return JsonResponse(
        {
            "ok": True,
            "intent": resp.intent,
            "text": resp.text,
            "items": resp.items,
            "sentiment": resp.sentiment or user_sentiment,
            "review_analysis": resp.review_analysis,
            "typing_ms": resp.typing_ms,
            "extras": resp.extras,
            "ai_enabled": bool(provider),
            "ai_provider": provider,
            "use_ai": bool(use_ai),
        }
    )


@require_http_methods(["GET"])
def admin_dashboard(request):
    """Analytics dashboard for chatbot and review checks."""
    chat_total = chat_user = chat_bot = 0
    review_total = review_fake = 0
    recent_chats = []
    recent_reviews = []
    intent_counts = {}

    if ChatLog is not None:
        chat_total = ChatLog.objects.count()
        chat_user = ChatLog.objects.filter(role="user").count()
        chat_bot = ChatLog.objects.filter(role="bot").count()
        recent_chats = list(ChatLog.objects.all()[:20])
        from django.db.models import Count

        for row in ChatLog.objects.exclude(intent="").values("intent").annotate(c=Count("id")).order_by("-c")[:8]:
            intent_counts[row["intent"]] = row["c"]

    if ReviewCheckLog is not None:
        review_total = ReviewCheckLog.objects.count()
        review_fake = ReviewCheckLog.objects.filter(label="fake").count()
        recent_reviews = list(ReviewCheckLog.objects.all()[:15])

    return render(
        request,
        "recommender/admin_dashboard.html",
        {
            "chat_total": chat_total,
            "chat_user": chat_user,
            "chat_bot": chat_bot,
            "review_total": review_total,
            "review_fake": review_fake,
            "recent_chats": recent_chats,
            "recent_reviews": recent_reviews,
            "intent_counts": intent_counts,
        },
    )
