from datetime import datetime
from database import get_connection


def get_all_pairs() -> list[dict]:
    """Return all causal pairs from memory ordered by confidence."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, cause, effect, confidence, seen_count, lag_days, last_seen
        FROM casual_pairs
        ORDER BY confidence DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_top_pairs(limit: int = 10) -> list[dict]:
    """Return the top N highest confidence causal pairs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, cause, effect, confidence, seen_count, lag_days
        FROM casual_pairs
        ORDER BY confidence DESC, seen_count DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_predictions(limit: int = 20) -> list[dict]:
    """Return the most recent predictions fired by the agent."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, trigger_headline, predicted_event,
               confidence, timeframe_days, fired_at, outcome,
               window_close, confirmed_at
        FROM predictions
        ORDER BY fired_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_prediction(prediction: dict) -> int:
    """Save a prediction to the database. Returns the new prediction id.

    Two bugs fixed here vs the original version:
    1. timeframe_days was always None because infer()'s output dict uses
       the key "lag_hours", not "timeframe_days" -- there was no key of
       that name to read. Now derived from lag_hours.
    2. window_close was computed in infer() but never persisted, which
       meant there was no way to later check whether a prediction's
       window had passed (needed for auto-confirmation).
    """
    conn = get_connection()
    cursor = conn.cursor()

    lag_hours = prediction.get("lag_hours")
    timeframe_days = round(lag_hours / 24, 1) if lag_hours else prediction.get("timeframe_days")

    cursor.execute("""
        INSERT INTO predictions (
            trigger_headline, predicted_event,
            confidence, timeframe_days,
            casual_pair_id, fired_at, window_close
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction.get("trigger_headline"),
        prediction.get("predicted_event"),
        prediction.get("confidence"),
        timeframe_days,
        prediction.get("causal_pair_id"),
        datetime.utcnow().isoformat(),
        prediction.get("window_close")
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    print(f"[MEMORY] Prediction saved with id: {new_id}")
    return new_id


def get_expired_pending_predictions() -> list[dict]:
    """Return predictions that are still marked 'pending' but whose
    window_close has already passed -- these are ready to be checked
    against the live web and confirmed or expired.

    Predictions with a NULL window_close (saved before the migration, or
    from a run where window_close wasn't set) are excluded, since there's
    no way to know if their window has passed.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, trigger_headline, predicted_event, confidence,
               timeframe_days, fired_at, window_close, casual_pair_id
        FROM predictions
        WHERE outcome = 'pending'
          AND window_close IS NOT NULL
          AND window_close <= ?
        ORDER BY window_close ASC
    """, (datetime.utcnow().isoformat(),))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_prediction_outcome(prediction_id: int, outcome: str) -> None:
    """Mark a prediction as 'confirmed' or 'expired' after checking the
    live web. Also stamps confirmed_at so the Alert Log can show when
    verification happened, not just when the prediction originally fired."""
    if outcome not in ("confirmed", "expired", "pending"):
        raise ValueError(f"Invalid outcome: {outcome}")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE predictions
        SET outcome = ?, confirmed_at = ?
        WHERE id = ?
    """, (outcome, datetime.utcnow().isoformat(), prediction_id))
    conn.commit()
    conn.close()
    print(f"[MEMORY] Prediction {prediction_id} marked as {outcome}")


def save_headline(title, source=None, published_at=None, sentiment=None, score=None):
    """Insert a headline if we haven't seen this exact title before.
    Returns True if it was a new row, False if it was a duplicate (skipped)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM headlines WHERE title = ? LIMIT 1", (title,))
    if cursor.fetchone():
        conn.close()
        return False
    cursor.execute(
        "INSERT INTO headlines (title, source, published_at, sentiment, score) VALUES (?, ?, ?, ?, ?)",
        (title, source, published_at, sentiment, score)
    )
    conn.commit()
    conn.close()
    return True

def get_memory_stats() -> dict:
    """Return high level stats about what the agent has learned."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM casual_pairs")
    total_pairs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM headlines")
    total_headlines = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM casual_pairs
        WHERE confidence >= 0.75
    """)
    strong_pairs = cursor.fetchone()[0]

    conn.close()

    return {
        "total_pairs":       total_pairs,
        "total_headlines":   total_headlines,
        "total_predictions": total_predictions,
        "strong_pairs":      strong_pairs
    }
