"""
chat.py — CausalPulse conversational layer.

Design: this is "RAG-lite," not a vector-search RAG pipeline. We don't need
embeddings for retrieval here because casual_engine.py already computes
sentence-transformer embeddings when it builds pairs — reusing that would mean
loading the model again just to answer a chat message, which is wasteful for
a hackathon deploy. Instead we do fast keyword/entity overlap against
get_top_pairs() to decide relevance, and only call Qwen once per message.

Flow:
  1. Pull top pairs from memory (memory.py) — cheap, no ML model load.
  2. Score overlap between the user's message and each pair's cause/effect text.
  3. If overlap clears a threshold, inject those pairs as grounding context.
  4. Call Qwen with a system prompt that enforces the CausalPulse voice and
     tells it whether it has grounding context or is answering generally.
  5. Return the reply plus which pair IDs (if any) were cited, so the frontend
     can highlight matching nodes in the Memory Bank D3 graph.
"""

import os
import re
import httpx
from memory import get_top_pairs
from tavily import TavilyClient

QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_KEY)

# Words that signal the person wants CURRENT/LIVE information, not a
# general causal explanation. Mirrors the same distinction casual_engine.py
# draws with urgency, just applied to conversational intent instead.
LIVE_QUERY_SIGNALS = {
    "now", "today", "currently", "current", "right now", "at the moment",
    "this week", "latest", "recent", "recently", "live", "price of",
    "how much is", "what is the price", "trading at", "worth now",
}


def _is_live_query(message: str) -> bool:
    """True if the message is asking about the present moment (a live
    price, today's news, etc) rather than a general 'why does X affect Y'
    causal question. Simple substring check on purpose: this only needs
    to catch the common phrasings, not be exhaustive — a false negative
    just means the person gets a general answer instead of a web search,
    which is a safe fallback, not a broken experience."""
    text = message.lower()
    return any(signal in text for signal in LIVE_QUERY_SIGNALS)


def get_live_context(query: str) -> list[dict]:
    """Search the web for current information relevant to the question.
    Returns a list of {title, content, url} dicts, or [] on any failure
    (never raises — a failed live search should degrade to a general
    answer, not break the chat entirely)."""
    try:
        result = tavily.search(query=query, max_results=3, search_depth="basic")
        return [
            {
                "title": r.get("title", ""),
                "content": r.get("content", "")[:300],
                "url": r.get("url", ""),
            }
            for r in result.get("results", [])[:3]
        ]
    except Exception as e:
        print(f"[CHAT] Tavily live search failed: {e}")
        return []

# Tune this: higher = stricter about what counts as "relevant enough to cite"
# Lowered from 2 to 1: with stemming below, a single matched root word (e.g.
# "interest") is already meaningful signal against a finance-headline corpus.
# Requiring 2 was silently rejecting real matches like "rate hikes" vs
# "raises interest rates" because "hikes" != "raises" as raw strings.
RELEVANCE_MIN_OVERLAP_WORDS = 1

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "did", "does", "do",
    "why", "what", "how", "when", "will", "would", "could", "and", "or",
    "to", "of", "in", "on", "for", "with", "that", "this", "it", "its",
    "as", "by", "at", "be", "been", "has", "have", "had",
}


def _stem(word: str) -> str:
    """Very naive stemmer: strip common suffixes so 'rates' matches 'rate'
    and 'increases' matches 'increase'. This is NOT linguistically correct
    (won't handle irregulars), but it's fast, dependency-free, and good
    enough for keyword overlap scoring on financial vocabulary."""
    for suffix in ("ing", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: -len(suffix)]
    return word


def _keywords(text: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords, stem. Simple on
    purpose — this only needs to be good enough to rank ~20-100 candidate
    pairs, not do real NLP. Save the heavy lifting for casual_engine.py."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 2}


def find_relevant_pairs(user_message: str, limit: int = 3, pool_size: int = 60) -> list[dict]:
    """Score memory pairs against the user's message by keyword overlap.

    Returns the top `limit` pairs whose overlap score is above threshold,
    sorted by (overlap_score, confidence) descending. Empty list means
    "no grounding found, let Qwen answer generally."
    """
    candidate_pairs = get_top_pairs(limit=pool_size)
    query_words = _keywords(user_message)
    if not query_words:
        return []

    scored = []
    for pair in candidate_pairs:
        pair_text = f"{pair.get('cause', '')} {pair.get('effect', '')}"
        pair_words = _keywords(pair_text)
        overlap = query_words & pair_words
        if len(overlap) >= RELEVANCE_MIN_OVERLAP_WORDS:
            scored.append((len(overlap), pair.get("confidence", 0), pair))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _, _, p in scored[:limit]]


def _build_system_prompt(grounded_pairs: list[dict], live_context: list[dict] | None = None) -> str:
    base_voice = (
        "You are the CausalPulse Assistant. CausalPulse's philosophy: "
        "'Causality is more fundamental than time, and time may arise out of "
        "causality.' Speak in cause-and-effect framing where it's natural, "
        "not forced into every sentence. Never use em dashes. This is a "
        "financial news causality tool, not a generic chatbot.\n\n"
        "Match your reply length to the message. A greeting gets a short, "
        "warm greeting back, not a lecture on causality. A casual or vague "
        "question gets 1-3 sentences. Save longer, structured answers for "
        "when the person asks something that actually requires depth (a "
        "specific causal question, a request to trace a chain, etc). "
        "Being concise is part of the brand voice, not a fallback."
    )

    live_block = ""
    if live_context:
        live_lines = [
            f"- {c['title']}: {c['content']}" for c in live_context
        ]
        live_block = (
            "\n\nCausalPulse just searched the live web for this question "
            "and found:\n" + "\n".join(live_lines) + "\n\nUse this current "
            "information to answer directly (e.g. state the actual current "
            "price/value if asked), then, where it fits, connect it to a "
            "likely causal effect. Do not say you lack real-time data if "
            "live search results are provided above; you have them."
        )

    if not grounded_pairs:
        return (
            base_voice
            + " You do not have a specific detected causal chain for this "
            "question. Answer generally from your own knowledge, but stay "
            "in the cause-and-effect voice, and do not imply this came from "
            "CausalPulse's memory bank."
            + live_block
        )

    context_lines = []
    for p in grounded_pairs:
        context_lines.append(
            f"- Cause: {p.get('cause')} -> Effect: {p.get('effect')} "
            f"(confidence: {p.get('confidence'):.2f}, "
            f"seen {p.get('seen_count', '?')} times, "
            f"lag: {p.get('lag_days', '?')} days)"
        )
    context_block = "\n".join(context_lines)

    return (
        base_voice
        + " CausalPulse has detected the following causal pattern(s) in its "
        "memory bank that are relevant to this question:\n"
        + context_block
        + "\n\nGround your answer in this data when it fits. Cite the "
        "specific confidence/seen-count numbers naturally in your answer. "
        "If the detected pairs don't fully answer the question, say so and "
        "add general context."
        + live_block
    )


async def get_chat_response(message: str, history: list[dict] | None = None) -> dict:
    """Main entry point, called from the /api/chat route.

    history: list of {"role": "user"|"assistant", "content": str}, most
    recent last. Keep this short (last ~6 messages) — we're not persisting
    chat history server-side for the hackathon build, the frontend just
    replays what it has in session state.
    """
    history = history or []
    grounded_pairs = find_relevant_pairs(message)

    live_context = []
    if _is_live_query(message):
        live_context = get_live_context(message)

    system_prompt = _build_system_prompt(grounded_pairs, live_context)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            QWEN_API_URL,
            headers={
                "Authorization": f"Bearer {QWEN_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen-plus",  # swap for whatever model your other Qwen calls use
                "messages": messages,
                "temperature": 0.4,
            },
        )
        response.raise_for_status()
        data = response.json()

    reply = data["choices"][0]["message"]["content"]
    # Qwen doesn't reliably follow the "never use em dashes" instruction in
    # the system prompt alone, so enforce it here too. A naive replace
    # left "word , word" (a space before the comma) since em dashes are
    # usually written with no surrounding spaces in the source text but
    # sometimes with spaces either side. Handle both by stripping any
    # spaces immediately around the dash before inserting the comma.
    reply = re.sub(r"\s*[—–]\s*", ", ", reply)
    reply = re.sub(r"\s*--\s*", ", ", reply)
    # Clean up any double punctuation that can result (e.g. ", ," -> ",")
    reply = re.sub(r",\s*,", ",", reply)

    return {
        "reply": reply,
        "cited_pairs": [
            {
                "id": p.get("id"),
                "cause": p.get("cause"),
                "effect": p.get("effect"),
                "confidence": p.get("confidence"),
            }
            for p in grounded_pairs
        ],
        "live_sources": [
            {"title": c["title"], "url": c["url"]} for c in live_context
        ],
    }