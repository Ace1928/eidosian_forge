#!/usr/bin/env python3
"""
Moltbook UI Dashboard.
Built with FastAPI, Jinja2, Tailwind, and HTMX.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from moltbook_forge.client import MoltbookClient, MockMoltbookClient, MoltbookUser
from moltbook_forge.interest import InterestEngine

# Configuration
MOCK_MODE = os.getenv("MOLTBOOK_MOCK", "true").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if MOCK_MODE:
        app.state.client = MockMoltbookClient()
    else:
        app.state.client = MoltbookClient()
    app.state.engine = InterestEngine()
    yield
    # Shutdown
    await app.state.client.close()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="moltbook_forge/ui/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    client: MoltbookClient = request.app.state.client
    engine: InterestEngine = request.app.state.engine
    
    posts = await client.get_posts(limit=50)
    me = await client.get_me()
    if me is None:
        me = MoltbookUser(
            name="EidosianForge",
            description="Profile unavailable",
            follower_count=0,
            following_count=0,
            karma=0,
        )
    ranked_posts = engine.rank_posts(posts)
    
    # Extract top activity
    top_activity = [p for p, s in ranked_posts[:5]]
    
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "posts": posts,
            "me": me,
            "top_activity": top_activity,
            "mock_mode": MOCK_MODE,
        },
    )

@app.get("/post/{post_id}", response_class=HTMLResponse)
async def post_detail(request: Request, post_id: str):
    client: MoltbookClient = request.app.state.client
    comments = await client.get_comments(post_id)
    return templates.TemplateResponse(
        request,
        "partials/comments.html",
        {"comments": comments},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
