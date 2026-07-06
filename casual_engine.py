import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from database import get_connection

load_dotenv()

# ── API CONFIGURATION ──────────────────────────────────────────
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL   = "qwen-turbo"
TAVILY_KEY   = os.getenv("TAVILY_API_KEY")

# ── MODELS ─────────────────────────────────────────────────────
# Loads once, reused across all calls
embedder     = SentenceTransformer("all-MiniLM-L6-v2")
tavily       = TavilyClient(api_key=TAVILY_KEY)

# ── DISPLAY FILTERS ────────────────────────────────────────────
MIN_CONFIDENCE    = 0.70   # minimum confidence to surface a prediction
MAX_PREDICTIONS   = 5      # maximum predictions shown at once
MIN_WORD_OVERLAP  = 3      # minimum semantic similarity threshold
SEMANTIC_DEDUP_POOL_SIZE = 500  # how many existing pairs store_pairs() embeds
                                 # and compares against per cycle -- bounds
                                 # the cost of semantic reinforcement matching
                                 # against your full 8000+ pair memory
HEADLINE_MATCH_THRESHOLD = 0.5  # minimum cosine similarity between a headline
                                 # and a known cause before it can trigger a
                                 # prediction. Raised from 0.35 -- that was
                                 # loose enough for vague/descriptive headlines
                                 # ("Exploring High Growth Tech Stocks...") to
                                 # match a cause concept despite not describing
                                 # an actual event.


# ── HELPERS ────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Calculate semantic similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_urgency(window_close: datetime) -> str:
    """Calculate urgency based on time remaining in window."""
    now = datetime.utcnow()
    hours_remaining = (window_close - now).total_seconds() / 3600

    if hours_remaining <= 6:
        return "HIGH"
    elif hours_remaining <= 48:
        return "MEDIUM"
    else:
        return "LOW"


def is_finance_relevant(headline: str) -> bool:
    """Filter out non-financial headlines."""
    keywords = [
        "stock", "market", "fed", "inflation", "rate", "oil",
        "nasdaq", "earnings", "gdp", "recession", "bank", "trade",
        "dollar", "crypto", "bond", "yield", "merger", "acquisition",
        "ipo", "shares", "equity", "commodity", "gold", "energy",
        "interest", "economy", "financial", "investment", "fund",
        "quarter", "revenue", "profit", "loss", "debt", "capital"
    ]
    headline_lower = headline.lower()
    return any(k in headline_lower for k in keywords)


# Phrasing patterns typical of listicles, evergreen guides, and general
# "exploring the space" analysis pieces rather than actual news events.
# These pass is_finance_relevant() easily (they're full of finance
# keywords) but describe no specific cause -- "Exploring High Growth
# Tech Stocks In The US Market" doesn't cause anything, unlike "Fed
# raises rates" or "Company reports strong earnings". Left unfiltered,
# these can still semantically match a known cause embedding closely
# enough to trigger a bogus prediction.
LISTICLE_PATTERNS = [
    "exploring", "top 10", "top 5", "best stocks", "best etfs",
    "guide to", "things to know", "everything you need",
    "why you should", "how to invest", "beginner's guide",
    "ultimate guide", "stocks to watch", "stocks to buy",
    "reasons to", "ways to", "tips for", "explained:",
]


def is_listicle_or_evergreen(headline: str) -> bool:
    """True if this reads like a listicle/guide/analysis piece rather
    than a discrete news event with an actual cause behind it."""
    headline_lower = headline.lower()
    return any(p in headline_lower for p in LISTICLE_PATTERNS)


# ── CORE FUNCTIONS ─────────────────────────────────────────────

def extract_pairs(headlines: list[str]) -> list[dict]:
    """
    Send a window of headlines to Qwen.
    Returns structured causal pairs with time awareness.
    """
    if not headlines:
        return []

    # Filter to finance relevant only before sending to Qwen
    relevant = [
        h for h in headlines
        if is_finance_relevant(h) and not is_listicle_or_evergreen(h)
    ]
    if not relevant:
        print("[ENGINE] No finance-relevant headlines found")
        return []

    headline_block = "\n".join(
        f"{i+1}. {h}" for i, h in enumerate(relevant)
    )

    prompt = f"""You are a financial causality detector.

Given these headlines, extract the underlying causal relationships.
Do NOT use the full headline text as cause or effect.
Extract the CORE CONCEPT behind each headline in maximum 8 words.

Return ONLY a JSON array. No explanation. No markdown. Just the array.

Each object must have exactly these keys:
- "cause": core concept of triggering event (max 8 words)
- "effect": core concept of resulting event (max 8 words)
- "confidence": one of "high", "medium", or "low"
- "lag_hours": estimated hours between cause and effect (integer)
- "window_hours": how long the effect opportunity typically lasts (integer)
- "urgency": one of "HIGH", "MEDIUM", or "LOW"

Example:
Headline: "Fed signals aggressive rate hike amid rising inflation"
cause: "central bank raises interest rates"
effect: "bond yields increase sharply"
confidence: "high"
lag_hours: 4
window_hours: 24
urgency: "HIGH"

Headlines:
{headline_block}

JSON array:"""

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=body)
        response.raise_for_status()
        raw  = response.json()
        text = raw["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        pairs = json.loads(text)
        print(f"[ENGINE] Extracted {len(pairs)} causal pairs")
        return pairs

    except Exception as e:
        print(f"[ENGINE] Qwen extraction failed: {e}")
        return []


def store_pairs(pairs: list[dict]):
    """
    Save extracted causal pairs to SQLite.
    Reinforce existing pairs, insert new ones.

    Reinforcement now uses semantic similarity instead of exact string
    matching. The original version only reinforced a pair if Qwen
    produced the EXACT same cause/effect wording as an existing row --
    but Qwen doesn't phrase the same underlying relationship identically
    across cycles ("central bank raises interest rates" vs "fed hikes
    rates"), so almost every live-fetched pair was silently creating a
    new row with seen_count=1 instead of reinforcing the real pattern.
    This is why "pattern match" counts stayed at 1 for most predictions
    generated from live headlines, even though the bootstrap data (from
    one large static CSV batch) happened to have real repeats.

    Cost tradeoff: this embeds existing pairs to compare against, which
    is more expensive than a SQL exact match. Capped via
    SEMANTIC_DEDUP_POOL_SIZE (comparing against your top-confidence
    pairs, not all 8000+) to keep this from noticeably slowing down
    every cycle -- tune that constant if match quality vs speed needs
    adjusting.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    confidence_map = {"high": 0.85, "medium": 0.60, "low": 0.35}
    SIMILARITY_THRESHOLD = 0.88  # how close cause->effect text must be to count as "the same pattern"

    # Load a bounded pool of existing pairs to compare against, ranked by
    # confidence (the pairs most likely to be worth reinforcing anyway).
    cursor.execute("""
        SELECT id, cause, effect, seen_count, confidence
        FROM casual_pairs
        ORDER BY confidence DESC, seen_count DESC
        LIMIT ?
    """, (SEMANTIC_DEDUP_POOL_SIZE,))
    existing = [dict(r) for r in cursor.fetchall()]

    if existing:
        existing_texts = [f"{r['cause']} -> {r['effect']}" for r in existing]
        existing_embeddings = embedder.encode(existing_texts)
    else:
        existing_embeddings = np.zeros((0, 384))  # all-MiniLM-L6-v2 output dim

    for pair in pairs:
        cause       = pair.get("cause", "").strip()
        effect      = pair.get("effect", "").strip()
        confidence  = pair.get("confidence", "medium")
        lag_hours   = pair.get("lag_hours", 24)

        if not cause or not effect:
            continue

        confidence_score = confidence_map.get(confidence, 0.60)
        combined_text = f"{cause} -> {effect}"
        new_embedding = embedder.encode(combined_text)

        match_idx = None
        if len(existing):
            norms = np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(new_embedding)
            norms[norms == 0] = 1e-9  # avoid divide-by-zero on any zero-vector edge case
            similarities = np.dot(existing_embeddings, new_embedding) / norms
            best_idx = int(np.argmax(similarities))
            if similarities[best_idx] >= SIMILARITY_THRESHOLD:
                match_idx = best_idx

        if match_idx is not None:
            match          = existing[match_idx]
            new_count      = match["seen_count"] + 1
            new_confidence = min(0.99, match["confidence"] + 0.02)
            cursor.execute("""
                UPDATE casual_pairs
                SET seen_count   = ?,
                    confidence   = ?,
                    last_seen    = ?
                WHERE id = ?
            """, (new_count, new_confidence,
                  datetime.utcnow().isoformat(), match["id"]))
            print(f"[ENGINE] Reinforced: '{cause}' → '{effect}' "
                  f"(matched existing '{match['cause']}' → '{match['effect']}', seen {new_count}x)")

            # Keep the in-memory pool in sync so multiple similar pairs
            # within the SAME batch also reinforce each other correctly,
            # instead of each one only ever matching the original DB state.
            existing[match_idx]["seen_count"] = new_count
            existing[match_idx]["confidence"] = new_confidence
        else:
            cursor.execute("""
                INSERT INTO casual_pairs
                    (cause, effect, confidence, lag_days,
                     seen_count, last_seen)
                VALUES (?, ?, ?, ?, 1, ?)
            """, (cause, effect, confidence_score,
                  max(1, lag_hours) if lag_hours > 0 else 4,
                  datetime.utcnow().isoformat()))
            new_id = cursor.lastrowid
            print(f"[ENGINE] New pair: '{cause}' → '{effect}'")

            # Add to the in-memory pool too, so later pairs in this same
            # batch can match against it rather than each becoming its
            # own separate seen_count=1 row.
            existing.append({
                "id": new_id, "cause": cause, "effect": effect,
                "seen_count": 1, "confidence": confidence_score
            })
            existing_embeddings = np.vstack([existing_embeddings, new_embedding])

    conn.commit()
    conn.close()


def get_tavily_context(headline: str) -> str:
    """
    Fetch real-time context for a headline using Tavily.
    Returns a brief summary with sources.
    """
    try:
        result  = tavily.search(query=headline, max_results=3)
        sources = result.get("results", [])
        context = " | ".join([
            f"{s['title']}: {s['content'][:150]}"
            for s in sources[:3]
        ])
        return context
    except Exception as e:
        print(f"[ENGINE] Tavily failed: {e}")
        return ""


def infer(today_headlines: list[str]) -> list[dict]:
    """
    Semantic inference engine.
    Matches today's headlines against stored causal memory.
    Filters and ranks by confidence, urgency and time window.
    Returns maximum 5 actionable predictions.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, cause, effect, confidence, lag_days, seen_count
        FROM casual_pairs
        ORDER BY confidence DESC
    """)
    known_pairs = cursor.fetchall()
    conn.close()

    if not known_pairs:
        print("[ENGINE] No causal pairs in memory yet")
        return []

    # Filter to finance relevant headlines only
    relevant_headlines = [
        h for h in today_headlines
        if is_finance_relevant(h) and not is_listicle_or_evergreen(h)
    ]

    if not relevant_headlines:
        print("[ENGINE] No finance-relevant headlines today")
        return []

    # Embed all causes from memory
    causes      = [pair["cause"] for pair in known_pairs]
    cause_embeddings = embedder.encode(causes)

    predictions = []
    seen_effects  = set()  # deduplicate same effect
    seen_triggers = set()  # deduplicate same trigger headline

    for headline in relevant_headlines:
        headline_embedding = embedder.encode(headline)

        for i, pair in enumerate(known_pairs):
            similarity = cosine_similarity(
                headline_embedding, cause_embeddings[i]
            )

            # Only match if semantically similar enough
            if similarity < HEADLINE_MATCH_THRESHOLD:
                continue

            # Skip duplicate trigger headlines
            if headline in seen_triggers:
                continue

            # Skip duplicate effects using fuzzy match
            duplicate = False
            for seen_effect in seen_effects:
                # Check if first 3 words overlap
                new_words  = set(pair["effect"].lower().split()[:4])
                seen_words = set(seen_effect.lower().split()[:4])
                if len(new_words & seen_words) >= 2:
                    duplicate = True
                    break
            if duplicate:
                continue

            # Skip low confidence pairs
            if pair["confidence"] < MIN_CONFIDENCE:
                continue

            # Calculate time window
            lag_hours    = pair["lag_days"] * 24
            window_open  = datetime.utcnow() + timedelta(hours=lag_hours * 0.5)
            window_close = datetime.utcnow() + timedelta(hours=lag_hours * 1.5)
            urgency      = get_urgency(window_close)

            # Skip LOW urgency predictions
            if urgency == "LOW" and len(predictions) >= 3:
                continue

            # Fetch Tavily context for HIGH urgency only
            # to preserve API credits
            context = ""
            if urgency == "HIGH":
                context = get_tavily_context(headline)

            seen_effects.add(pair["effect"])
            seen_triggers.add(headline)
            predictions.append({
                "trigger_headline": headline,
                "predicted_event":  pair["effect"],
                "confidence":       round(pair["confidence"], 2),
                "similarity_score": round(similarity, 2),
                "lag_hours":        lag_hours,
                "window_open":      window_open.isoformat(),
                "window_close":     window_close.isoformat(),
                "urgency":          urgency,
                "seen_count":       pair["seen_count"],
                "causal_pair_id":   pair["id"],
                "context":          context
            })

    # Sort: urgency first, then confidence
    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    predictions.sort(key=lambda x: (
        urgency_order[x["urgency"]], -x["confidence"]
    ))

    # Return only top MAX_PREDICTIONS
    final = predictions[:MAX_PREDICTIONS]
    print(f"[ENGINE] {len(final)} actionable predictions surfaced")
    return final


def run_cycle(headlines: list[str]) -> list[dict]:
    """
    Full autonomous cycle:
    1. Filter relevant headlines
    2. Extract causal pairs via Qwen
    3. Store/reinforce memory
    4. Run semantic inference
    5. Save predictions to database
    6. Return filtered actionable predictions
    """
    print(f"[ENGINE] Starting cycle with {len(headlines)} headlines")
    pairs = extract_pairs(headlines)
    store_pairs(pairs)
    predictions = infer(headlines)

    # Save predictions to database
    from memory import save_prediction
    for p in predictions:
        save_prediction(p)

    return predictions


def _judge_prediction_outcome(prediction: dict, live_context: str) -> str:
    """
    Ask Qwen to judge whether a prediction's effect actually happened,
    given live web search results for that effect. Returns "confirmed"
    or "expired" -- never "pending", since this is only called on
    predictions whose window has already closed (a decision has to be
    made one way or the other).

    Defaults to "expired" (not "confirmed") whenever the evidence is
    insufficient or ambiguous. Auto-confirming without real evidence
    would quietly make the agent look more accurate than it is, which
    defeats the point of building a genuine confirmation loop.
    """
    if not live_context:
        # No web evidence at all -- can't responsibly claim confirmed.
        return "expired"

    prompt = f"""You are verifying whether a financial prediction came true.

Predicted effect: "{prediction['predicted_event']}"
Original trigger headline: "{prediction['trigger_headline']}"
Prediction window closed at: {prediction.get('window_close', 'unknown')}

Recent web search results about this predicted effect:
{live_context}

Based ONLY on the search results above, did the predicted effect actually
happen? Respond with ONLY a JSON object, no markdown, no explanation:
{{"outcome": "confirmed" or "expired", "reasoning": "one short sentence"}}

Rules:
- "confirmed" only if the search results clearly show the predicted effect occurred
- "expired" if the results are unclear, unrelated, or show it did NOT happen
- When in doubt, choose "expired" -- do not guess confirmed without clear evidence"""

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=body)
        response.raise_for_status()
        raw = response.json()
        text = raw["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        verdict = json.loads(text)
        outcome = verdict.get("outcome", "expired")
        reasoning = verdict.get("reasoning", "")
        print(f"[ENGINE] Verdict for '{prediction['predicted_event']}': {outcome} -- {reasoning}")
        return outcome if outcome in ("confirmed", "expired") else "expired"
    except Exception as e:
        print(f"[ENGINE] Confirmation judgment failed: {e}")
        return "expired"


def confirm_predictions() -> dict:
    """
    Check every prediction whose window has closed but is still marked
    'pending', search the live web for evidence of whether the predicted
    effect actually happened, and update its outcome accordingly.

    This closes the loop the Alert Log always implied but never actually
    delivered on: predictions could show 'pending' or 'expired' before,
    but nothing ever set outcome to 'confirmed', and nothing ever
    transitioned a pending prediction to expired once its window passed.

    Intended to be called manually (via a dedicated endpoint) or folded
    into the scheduler's periodic cycle later.
    """
    from memory import get_expired_pending_predictions, update_prediction_outcome

    candidates = get_expired_pending_predictions()
    if not candidates:
        print("[ENGINE] No pending predictions past their window to confirm")
        return {"checked": 0, "confirmed": 0, "expired": 0, "details": []}

    print(f"[ENGINE] Confirming {len(candidates)} expired-window predictions")

    checked = 0
    confirmed = 0
    expired = 0
    details = []

    for pred in candidates:
        context = get_tavily_context(pred["predicted_event"])
        outcome = _judge_prediction_outcome(pred, context)
        update_prediction_outcome(pred["id"], outcome)

        checked += 1
        if outcome == "confirmed":
            confirmed += 1
        else:
            expired += 1

        details.append({
            "id": pred["id"],
            "predicted_event": pred["predicted_event"],
            "outcome": outcome
        })

    result = {
        "checked": checked,
        "confirmed": confirmed,
        "expired": expired,
        "details": details
    }
    print(f"[ENGINE] Confirmation pass complete: {result['confirmed']} confirmed, {result['expired']} expired")
    return result