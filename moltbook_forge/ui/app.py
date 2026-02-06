#!/usr/bin/env python3
"""
Moltbook UI Dashboard - Nexus Phase 4.
Robust, observability-enabled, and relational.
"""

import os
import time
import logging
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
from moltbook_forge.engagement import EngagementEngine

# Configuration
MOCK_MODE = os.getenv("MOLTBOOK_MOCK", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("NexusUI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """System-wide lifecycle management."""
    app.state.diag = DiagnosticsForge(service_name="nexus_ui")
    app.state.diag.log_event("info", "Nexus UI Initializing", mode="MOCK" if MOCK_MODE else "REAL")
    
    if MOCK_MODE:
        app.state.client = MockMoltbookClient()
    else:
        app.state.client = MoltbookClient()
        
    app.state.engine = InterestEngine()
    app.state.graph = SocialGraph()
    app.state.engagement = EngagementEngine()
    
    yield
    # Shutdown
    await app.state.client.close()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="moltbook_forge/ui/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main Eidosian Command Center dashboard."""
    start_time = time.time()
    client: MoltbookClient = request.app.state.client
    engine: InterestEngine = request.app.state.engine
    graph: SocialGraph = request.app.state.graph
    diag: DiagnosticsForge = request.app.state.diag
    
    try:
        # Fetch current signal stream
        posts = await client.get_posts(limit=25)
        me = await client.get_me()
        if me is None:
            me = MoltbookUser(name="EidosianForge", description="Identity Cache Miss", follower_count=0, following_count=0, karma=0)
        
        # Analyze and score
        analyzed_posts = [(p, engine.analyze_post(p)) for p in posts]
        analyzed_posts.sort(key=lambda x: x[1].total, reverse=True)
        
        # Dynamic Graph Mapping: Link Eidos to every agent in the current feed
        for p in posts:
            graph.add_link(me.name, p.author)
            
        top_activity = [p for p, s in analyzed_posts[:5]]
        
        diag.log_event("info", "Dashboard Rendered", post_count=len(posts), latency=time.time() - start_time)
        
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
        logger.error(f"Critical UI Error: {e}", exc_info=True)
        diag.log_event("error", "Dashboard Error", error=str(e))
        return HTMLResponse(content=f"Nexus Critical Error: {e}", status_code=500)

@app.get("/api/stats")
async def get_stats(request: Request):
    """System health and heuristic filtering metrics."""
    start_time = time.time()
    engine: InterestEngine = request.app.state.engine
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
    """Fetch the relational agent graph."""
    start_time = time.time()
    graph: SocialGraph = request.app.state.graph
    return NexusResponse.ok(graph.get_graph(), start_time)

@app.get("/api/synthesize/{post_id}", response_class=HTMLResponse)
async def synthesize_response(request: Request, post_id: str):
    """Generate agentic engagement draft."""
    client: MoltbookClient = request.app.state.client
    engine: InterestEngine = request.app.state.engine
    engagement: EngagementEngine = request.app.state.engagement
    
    posts = await client.get_posts(limit=50)
    target_post = next((p for p in posts if p.id == post_id), None)
    
    if not target_post:
        return HTMLResponse(content="Signal lost or invalid ID.", status_code=404)
        
    analysis = engine.analyze_post(target_post)
    draft = await engagement.draft_response(target_post, analysis)
    
    return HTMLResponse(content=f'<div class="p-4 bg-teal-500/5 border border-teal-500/20 rounded-lg text-xs italic font-mono text-teal-200 animate-pulse">{draft}</div>')

@app.post("/reputation/{username}/{delta}", response_class=HTMLResponse)
async def update_reputation(request: Request, username: str, delta: float):
    """Manual reputation adjustment via UI."""
    engine: InterestEngine = request.app.state.engine
    engine.update_reputation(username, delta)
    new_rep = engine.calculate_reputation(username)
    return HTMLResponse(content=f'<span class="text-[10px] text-amber-400 font-bold tracking-tighter">TRUST: {new_rep}</span>')

@app.get("/post/{post_id}", response_class=HTMLResponse)
async def post_detail(request: Request, post_id: str):
    """Fetch threaded signal engagement (comments)."""
    client: MoltbookClient = request.app.state.client
    comments = await client.get_comments(post_id)
    return templates.TemplateResponse(
        request,
        "partials/comments.html",
        {"comments": comments},
    )

if __name__ == "__main__":
    import uvicorn
    # High-performance uvicorn worker
    uvicorn.run("moltbook_forge.ui.app:app", host="0.0.0.0", port=8080, reload=False, workers=1)