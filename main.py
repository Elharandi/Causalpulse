import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

from database import init_db
from scheduler import start_scheduler
from api.routes import router

# Global scheduler reference
scheduler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    global scheduler

    # Startup
    print("[MAIN] CasualPulse starting up...")
    init_db()
    scheduler = start_scheduler(interval_minutes=120)
    print("[MAIN] All systems running")

    yield  # App is now running and serving requests

    # Shutdown
    print("[MAIN] Shutting down...")
    if scheduler:
        scheduler.shutdown()
    print("[MAIN] Goodbye")


app = FastAPI(
    title="CasualPulse",
    description="Autonomous financial news causality agent",
    version="1.0.0",
    lifespan=lifespan
)

# Allow frontend to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount API routes under /api
app.include_router(router, prefix="/api")

# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)