import sqlite3
import os
from datetime import datetime
DB_PATH = os.getenv("DB_PATH", "casualpulse.db")

def get_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable accessing columns by name
    return conn

def init_db():
    """Create all tables if they don't exist yet."""
    conn = get_connection()
    cursor = conn.cursor()


    # Table one : every headline the aget reads
    cursor.execute(""" 
     CREATE TABLE IF NOT EXISTS headlines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        source TEXT,
        published_at TEXT,
        sentiment TEXT,
        score REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                   
    )
""")
    # Table two : every casual pair the agent sees
    cursor.execute("""
     CREATE TABLE IF NOT EXISTS casual_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cause TEXT NOT NULL,
        effect TEXT NOT NULL,
        confidence REAL DEFAULT 0.5,
        seen_count INTEGER DEFAULT 1,
        lag_days INTEGER DEFAULT 1,
        last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")

    # Table three : every prediction the agent makes

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trigger_headline TEXT NOT NULL,
        predicted_event TEXT NOT NULL,
        confidence REAL,
        timeframe_days INTEGER,
        casual_pair_id INTEGER,
        fired_at TEXT DEFAULT CURRENT_TIMESTAMP,
        outcome TEXT DEFAULT 'pending',
        FOREIGN KEY (casual_pair_id) REFERENCES casual_pairs(id)
        )
""")

    # Migration: predictions was originally created without window_close,
    # so infer()'s computed window could never be persisted or checked
    # later for auto-confirmation. Add it if missing, on existing DBs too.
    # Also add confirmed_at so the Alert Log can show when a prediction
    # was actually verified, not just when it fired.
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "window_close" not in existing_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN window_close TEXT")
        print("[DB] Migrated: added window_close column to predictions")
    if "confirmed_at" not in existing_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN confirmed_at TEXT")
        print("[DB] Migrated: added confirmed_at column to predictions")

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at: {DB_PATH}")

if __name__ == "__main__":
    init_db()
