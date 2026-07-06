import os
import csv
import time
from dotenv import load_dotenv
from database import init_db, get_connection
from casual_engine import extract_pairs, store_pairs

load_dotenv()

DATA_PATH    = os.path.join("data", "clean_headlines.csv")
BATCH_SIZE   = 20   # headlines per Qwen call
MAX_BATCHES  = 500  # safety cap — remove later if needed
SLEEP_BETWEEN = 2   # seconds between Qwen calls to avoid rate limiting


def load_headlines() -> list[dict]:
    """Load clean headlines from CSV."""
    headlines = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("title"):
                headlines.append({
                    "title":        row["title"],
                    "published_at": row.get("published_at", "")
                })
    print(f"[BOOTSTRAP] Loaded {len(headlines)} headlines from CSV")
    return headlines


def already_bootstrapped() -> bool:
    """Check if bootstrap has already run."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM casual_pairs")
    count = cursor.fetchone()[0]
    conn.close()
    return count > 50


def get_last_batch() -> int:
    """Return the last successfully processed batch number."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM casual_pairs")
    count = cursor.fetchone()[0]
    conn.close()
    # Estimate which batch we're on
    return max(0, (count // 16) - 1)


def run_bootstrap():
    """
    One-time bootstrap run.
    Feeds historical headlines to the causal engine in batches.
    No Tavily — saves API credits.
    No infer() — just extract and store pairs.
    """
    print("[BOOTSTRAP] Starting bootstrap...")
    init_db()

    headlines = load_headlines()
    total     = len(headlines)
    batches   = [
        headlines[i:i + BATCH_SIZE]
        for i in range(0, total, BATCH_SIZE)
    ]

    print(f"[BOOTSTRAP] {total} headlines → {len(batches)} batches of {BATCH_SIZE}")
    print(f"[BOOTSTRAP] Running {min(MAX_BATCHES, len(batches))} batches")

    # Resume from where we left off
    start_batch = get_last_batch()
    if start_batch > 0:
        print(f"[BOOTSTRAP] Resuming from batch ~{start_batch}...")
    
    print("[BOOTSTRAP] This will take a while. Do not interrupt.\n")

    success = 0
    failed  = 0

   # Resume from where we left off
    start_batch = get_last_batch()
    if start_batch > 0:
        print(f"[BOOTSTRAP] Resuming from batch {start_batch}...")

    success = 0
    failed  = 0
    
    for i, batch in enumerate(batches[start_batch:MAX_BATCHES]):
        actual_batch = i + start_batch
        titles = [h["title"] for h in batch]
        print(f"[BOOTSTRAP] Batch {actual_batch+1}/{min(MAX_BATCHES, len(batches))}...")

        try:
            pairs = extract_pairs(titles)
            store_pairs(pairs)
            success += 1
        except Exception as e:
            print(f"[BOOTSTRAP] Batch {actual_batch+1} failed: {e}")
            failed += 1

        time.sleep(SLEEP_BETWEEN)

        if (actual_batch + 1) % 10 == 0:
            from memory import get_memory_stats
            stats = get_memory_stats()
            print(f"\n[BOOTSTRAP] Progress: {stats['total_pairs']} pairs learned so far\n")
    print(f"\n[BOOTSTRAP] Complete!")
    print(f"[BOOTSTRAP] Batches: {success} success, {failed} failed")

    from memory import get_memory_stats
    stats = get_memory_stats()
    print(f"[BOOTSTRAP] Final memory: {stats}")


if __name__ == "__main__":
    run_bootstrap()