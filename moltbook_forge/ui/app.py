#!/usr/bin/env python3
"""
Moltbook UI Dashboard.
Built with FastAPI, Jinja2, Tailwind, and HTMX.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from diagnostics_forge.core import DiagnosticsForge
from moltbook_forge.client import MoltbookClient, MockMoltbookClient, MoltbookUser
from moltbook_forge.interest import InterestEngine, MARKERS
from moltbook_forge.ui.schemas import NexusResponse
from moltbook_forge.ui.graph_api import SocialGraph

# Configuration
MOCK_MODE = os.getenv("MOLTBOOK_MOCK", "true").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.diag = DiagnosticsForge(service_name="nexus_ui")
    app.state.diag.log_event(
        "info",
        "Initializing Moltbook Nexus UI",
        mode="MOCK" if MOCK_MODE else "REAL",
    )
    
    if MOCK_MODE:
        app.state.client = MockMoltbookClient()
    else:
        app.state.client = MoltbookClient()
    app.state.engine = InterestEngine()
    app.state.graph = SocialGraph()
    yield
    # Shutdown
    await app.state.client.close()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="moltbook_forge/ui/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    start_time = time.time()
    client: MoltbookClient = request.app.state.client
    engine: InterestEngine = request.app.state.engine
    diag: DiagnosticsForge = request.app.state.diag
    
    try:
        posts = await client.get_posts(limit=50)
        me = await client.get_me()
        if me is None:
            me = MoltbookUser(name="EidosianForge", description="Profile unavailable", follower_count=0, following_count=0, karma=0)
        
        analyzed_posts = [(p, engine.analyze_post(p)) for p in posts]
        analyzed_posts.sort(key=lambda x: x[1].total, reverse=True)
        top_activity = [p for p, s in analyzed_posts[:5]]
        
        diag.log_event(
            "info",
            "Dashboard rendered",
            posts=len(posts),
            latency=time.time() - start_time,
        )
        
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "posts": analyzed_posts,
                "me": me,
                "top_activity": top_activity,
                "mock_mode": MOCK_MODE,
            },
        )
    except Exception as e:
        diag.log_event("error", "Dashboard error", error=str(e))
        return HTMLResponse(content=f"Nexus Critical Error: {e}", status_code=500)

@app.get("/api/stats")
async def get_stats(request: Request):
    start_time = time.time()
    engine: InterestEngine = request.app.state.engine
    
    # Collective reputation stats
    trusted_count = len([r for r in engine.reputation_map.values() if r > 5])
    
    stats = {
        "trusted_agents": trusted_count,
        "active_filters": len(MARKERS),
        "mock_mode": MOCK_MODE,
        "system_status": "OPTIMAL"
    }
    
    return NexusResponse.ok(stats, start_time)

@app.get("/api/graph")
async def get_graph(request: Request):
    start_time = time.time()
    graph: SocialGraph = request.app.state.graph
    client: MoltbookClient = request.app.state.client
    
    # Lazily populate graph from recent posts if empty
    if not graph.nodes:
        posts = await client.get_posts(limit=10)
        for post in posts:
            graph.add_link("EidosianForge", post.author)
            
    return NexusResponse.ok(graph.get_graph(), start_time)

@app.post("/api/reputation/{username}/{delta}")
async def update_reputation_api(request: Request, username: str, delta: float):
    start_time = time.time()
    engine: InterestEngine = request.app.state.engine
    engine.update_reputation(username, delta)
    new_rep = engine.calculate_reputation(username)
    
    return NexusResponse.ok({
        "username": username,
        "delta": delta,
        "new_score": new_rep
    }, start_time)

@app.post("/reputation/{username}/{delta}", response_class=HTMLResponse)
async def update_reputation(request: Request, username: str, delta: float):
    engine: InterestEngine = request.app.state.engine
    engine.update_reputation(username, delta)
    new_rep = engine.calculate_reputation(username)
    return HTMLResponse(content=f'<span class="text-[10px] text-amber-400 font-bold">REP: {new_rep}</span>')

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
    uvicorn.run("moltbook_forge.ui.app:app", host="0.0.0.0", port=8080, reload=False)
