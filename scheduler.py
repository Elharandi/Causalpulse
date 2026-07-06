import os
import re
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
from tavily import TavilyClient

load_dotenv()

# Swapped from NewsAPI to Tavily: NewsAPI's free "Developer" tier is
# restricted to localhost use by their terms of service, so it would
# have silently broken (or risked getting blocked) once deployed to
# Alibaba Cloud ECS with a public IP. Tavily is already a trusted
# dependency elsewhere in the project (casual_engine.py's live-context
# lookups, chat.py's live search), so this reuses infrastructure
# instead of adding a new API key to manage.
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_KEY)

# Landed on this phrasing after comparing 4 candidates against real
# Tavily results -- this one returned genuinely causal, event-driven
# headlines (Fed policy shifts, oil price moves, jobs data) instead of
# repetitive "Stock market today: Dow slips" templates or navigation
# pages like "Yahoo Finance - Stock Market Live". See test_tavily_query.py.
#
# Tried splitting this into multiple topic-focused queries (oil, crypto,
# bonds, tech separately) for broader coverage, but that pulled in a lot
# of generic category/navigation pages ("Cryptocurrency News -
# Investopedia", "Commodities - Bloomberg", "META_TITLE_SECTORS") that
# don't match the ticker-symbol or listicle filters below since they're
# not tied to a specific ticker or "top 10" phrasing, just vague section
# titles. Reverted to the single well-tested query -- fewer headlines
# per cycle, but reliably real ones.
SEARCH_QUERY = "stock market fed policy inflation today"


# Patterns that indicate the result is a navigation page, video, ticker
# listing, or podcast episode rather than an actual news article --
# these pass is_finance_relevant()'s keyword check but produce weak or
# junk causal pairs when handed to Qwen. Checked case-insensitively.
LOW_QUALITY_PATTERNS = [
    "watch ",              # video titles, e.g. "Watch Antony Blinken on..."
    "stock forecast",      # single-ticker forecast pages
    "price target",        # single-ticker forecast pages
    "masters in business", # podcast series
    "stock price, news, quote", # Yahoo Finance ticker listing pages
    "charts, data & news", # ticker listing pages
]

# Listicle/evergreen-guide phrasing -- these describe no specific event
# ("Exploring High Growth Tech Stocks In The US Market") but are full of
# finance keywords, so they can still slip past relevance filters and
# even semantically match a known cause closely enough to trigger a
# bogus prediction downstream.
LISTICLE_PATTERNS = [
    "exploring", "top 10", "top 5", "best stocks", "best etfs",
    "guide to", "things to know", "everything you need",
    "why you should", "how to invest", "beginner's guide",
    "ultimate guide", "stocks to watch", "stocks to buy",
    "reasons to", "ways to", "tips for", "explained:",
]

# Bare ticker/futures symbol in parentheses, e.g. "(CL=F)", "(AAPL)",
# "(^DJI)", "(TASE.TA)" -- titles built around a raw quote, like "Crude
# Oil Aug 26 (CL=F) - Yahoo Finance", describe no event either. Same
# failure mode as listicles: keyword-rich, semantically vague, capable
# of matching a known cause and producing a nonsensical prediction
# (this is literally how "Crude Oil (CL=F) -> gold price retreats" got
# generated -- caught here now, filtered before it ever reaches the
# causal engine).
TICKER_SYMBOL_PATTERN = re.compile(r"\([A-Z0-9\^]{1,8}(=F|\.[A-Z]{1,3})?\)")


def _is_low_quality_headline(title: str) -> bool:
    """True if this looks like a navigation page, video, ticker listing,
    quote page, listicle, or podcast episode rather than a real news
    article describing an actual event."""
    title_lower = title.lower()
    if any(p in title_lower for p in LOW_QUALITY_PATTERNS):
        return True
    if any(p in title_lower for p in LISTICLE_PATTERNS):
        return True
    if TICKER_SYMBOL_PATTERN.search(title):
        return True
    return False


def fetch_headlines() -> list[str]:
    try:
        result = tavily.search(
            query=SEARCH_QUERY,
            topic="finance",
            max_results=20,
            days=1,
        )
        articles = result.get("results", [])
        headlines = [
            a["title"] for a in articles
            if a.get("title") and not _is_low_quality_headline(a["title"])
        ]

        # Persist to the headlines table -- covers both the scheduled
        # job and the manual "Run Cycle" button, since both call this
        # function. save_headline() dedupes by title so re-fetching the
        # same rolling 24h window doesn't inflate the count.
        from memory import save_headline
        new_count = sum(1 for h in headlines if save_headline(h))
        print(f"[SCHEDULER] Fetched {len(headlines)} headlines ({new_count} new, saved)")

        return headlines

    except Exception as e:
        print(f"[SCHEDULER] Tavily fetch failed: {e}")
        return []


def scheduled_job():
    """The job that runs automatically every X minutes."""
    print(f"[SCHEDULER] Running cycle at {datetime.utcnow().isoformat()}")
    from casual_engine import run_cycle
    headlines = fetch_headlines()
    if headlines:
        run_cycle(headlines)
    else:
        print("[SCHEDULER] No headlines fetched, skipping cycle")


def start_scheduler(interval_minutes: int = 120):
    """Start the background scheduler.

    Default changed from 30 to 120 minutes: fetch_headlines() searches a
    rolling 24-hour window (Tavily's days=1), so a 30-minute cadence was
    mostly re-fetching the same ~24 hours of content over and over,
    burning API calls for little new signal. 2 hours still keeps memory
    accumulating steadily; the "Run Cycle" button in the UI covers
    on-demand fetches for demos regardless of this interval.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_job,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id="causal_cycle",
        name="CasualPulse Causal Cycle",
        replace_existing=True
    )
    scheduler.start()
    print(f"[SCHEDULER] Started — running every {interval_minutes} minutes")
    return scheduler