from fastapi import APIRouter
from memory import get_all_pairs, get_top_pairs, get_predictions, get_memory_stats
from casual_engine import run_cycle
from scheduler import fetch_headlines
from pydantic import BaseModel
from chat import get_chat_response
from casual_engine import run_cycle, confirm_predictions
router = APIRouter()
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

@router.get("/stats")
def stats():
    """Return high level memory stats for the dashboard header."""
    return get_memory_stats()


@router.get("/pairs")
def pairs(limit: int = 20):
    """Return top causal pairs the agent has learned."""
    return get_top_pairs(limit)


@router.get("/predictions")
def predictions(limit: int = 20):
    """Return the most recent predictions the agent fired."""
    return get_predictions(limit)


@router.get("/pairs/all")
def all_pairs():
    """Return every causal pair in memory."""
    return get_all_pairs()


@router.post("/cycle")
def manual_cycle():
    """
    Manually trigger a fetch + causal cycle.
    Useful for demos — judges can press a button and watch it learn.
    """
    headlines = fetch_headlines()
    if not headlines:
        return {"status": "error", "message": "No headlines fetched"}

    predictions = run_cycle(headlines)
    return {
        "status": "ok",
        "headlines_processed": len(headlines),
        "predictions_generated": len(predictions),
        "predictions": predictions[:5]
    }


@router.post("/confirm")
def manual_confirm():
    """
    Manually trigger a confirmation pass: checks every prediction whose
    window has closed but is still 'pending', searches the live web for
    each one, and updates outcome to 'confirmed' or 'expired'.

    Useful for demos the same way /cycle is -- judges can press a button
    and watch pending predictions resolve into confirmed/expired in the
    Alert Log, instead of that column staying frozen forever.
    """
    result = confirm_predictions()
    return {
        "status": "ok",
        "checked": result["checked"],
        "confirmed": result["confirmed"],
        "expired": result["expired"],
        "details": result["details"]
    }

@router.get("/headlines/live")
def live_headlines():
    """Fetch and return latest headlines without running a full cycle."""
    headlines = fetch_headlines()
    return {
        "total": len(headlines),
        "headlines": headlines
    }






@router.post("/chat")
async def chat(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = await get_chat_response(request.message, history)
    return result



