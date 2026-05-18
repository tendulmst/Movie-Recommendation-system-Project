"""Optional OpenAI / Gemini integration for chat replies."""
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _post_json(url: str, headers: Dict[str, str], payload: Dict) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openai(
    message: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    if not api_key:
        return None
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-8:]:
        role = "assistant" if h.get("role") == "bot" else "user"
        messages.append({"role": role, "content": h.get("content", "")})
    messages.append({"role": "user", "content": message})
    try:
        body = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            {"model": model, "messages": messages, "max_tokens": 400, "temperature": 0.7},
        )
        return body["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("OpenAI call failed: %s", e)
        return None


def call_gemini(
    message: str,
    history: List[Dict[str, str]],
    system_prompt: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
) -> Optional[str]:
    if not api_key:
        return None
    parts = [{"text": system_prompt}]
    for h in history[-8:]:
        prefix = "Assistant" if h.get("role") == "bot" else "User"
        parts.append({"text": f"{prefix}: {h.get('content', '')}"})
    parts.append({"text": f"User: {message}"})
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    try:
        body = _post_json(
            url,
            {"Content-Type": "application/json"},
            {"contents": [{"parts": parts}]},
        )
        return body["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("Gemini call failed: %s", e)
        return None


def maybe_enhance_reply(
    message: str,
    history: List[Dict[str, str]],
    local_text: str,
    intent: str,
) -> str:
    """Use external AI when configured; otherwise return local_text."""
    provider = (os.environ.get("CHAT_AI_PROVIDER") or "").strip().lower()
    if not provider:
        return local_text

    system = (
        "You are a movie recommendation assistant. Be concise. "
        f"The user's intent was detected as: {intent}. "
        "Use this local answer as ground truth for movie facts:\n" + local_text
    )

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        enhanced = call_openai(message, history, system, key, model)
        return enhanced or local_text

    if provider in ("gemini", "google"):
        key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        enhanced = call_gemini(message, history, system, key, model)
        return enhanced or local_text

    return local_text
