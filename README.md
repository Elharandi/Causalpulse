# CausalPulse

**An autonomous financial news causality agent.** CausalPulse continuously reads financial headlines, uses Qwen (via Alibaba Dashscope) to extract causal patterns between events (e.g. "Fed raises rates" → "tech stocks decline"), stores them in a growing memory bank, and uses that accumulated knowledge to predict future market events from new headlines as they arrive.

**Live demo:** http://47.253.210.122:8000

Built for the Qwen Cloud Global AI Hackathon (Track 1: MemoryAgent).

---

## How it works

1. **Ingest** — a background scheduler (APScheduler) pulls fresh financial headlines every 2 hours via Tavily's search API, filtering out low-quality results (ticker listings, listicles, navigation pages).
2. **Extract** — each headline is run through Qwen to extract core cause → effect concept pairs.
3. **Remember** — pairs are deduplicated using sentence-transformer cosine similarity and stored in a persistent memory bank (SQLite), with confidence scores that strengthen the more often a pattern is observed.
4. **Predict** — when a new headline arrives, it's matched against the memory bank; if a strong causal match is found, CausalPulse fires a prediction with a confidence score and expected time window.
5. **Verify** — once a prediction's time window closes, the agent checks the live web to confirm or expire it, closing the feedback loop.

## Tech stack

- **Backend**: FastAPI, SQLite, APScheduler
- **AI**: Qwen (Alibaba Dashscope, international endpoint), sentence-transformers (CPU-only PyTorch) for semantic deduplication
- **Live data**: Tavily search API
- **Frontend**: Single-file vanilla JS + D3.js (no build step)

## Project structure

```
casualpulse/
├── main.py              FastAPI app entrypoint (lifespan, static files, router)
├── database.py          SQLite schema + connection handling
├── casual_engine.py      Core engine: semantic matching, Qwen extraction, prediction logic
├── scheduler.py          Background job: fetches + persists headlines every 2 hours
├── memory.py             All read/write helpers for the memory bank
├── chat.py               RAG-lite chat endpoint, grounded in the memory bank
├── bootstrap.py          One-time script to seed the memory bank from a historical CSV
├── api/
│   └── routes.py         All API endpoints (/stats, /pairs, /predictions, /cycle, /chat, ...)
├── frontend/
│   └── index.html        Full single-page dashboard (Dashboard, Predictions, Chain Explorer,
│                          Memory Bank, Live Feed, Alert Log)
└── data/
    └── clean_headlines.csv   Historical headlines used only for the one-time bootstrap
```

## Running it locally

**Requirements:** Python 3.11+ (developed on 3.13.3)

```bash
git clone https://github.com/YOUR_USERNAME/causalpulse.git
cd causalpulse

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install fastapi uvicorn apscheduler python-dotenv tavily-python sentence-transformers requests
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env
# then edit .env and add your own API keys:
#   QWEN_API_KEY   - from Alibaba Dashscope (international endpoint)
#   TAVILY_API_KEY - from tavily.com
```

**Seed the memory bank** (optional but recommended — without this, predictions start from zero learned patterns):

```bash
python bootstrap.py
```

This reads `data/clean_headlines.csv` in batches and populates `casual_pairs`. It's resumable if interrupted.

**Run the app:**

```bash
python main.py
```

Then open `http://localhost:8000`.

## Notes on tables naming

Internally, database tables use "casual" rather than "causal" (e.g. `casual_pairs`, `casual_engine.py`) — a naming choice made early on and kept for consistency across the codebase rather than risk breaking references mid-build.

## License

MIT — see [LICENSE](LICENSE).
