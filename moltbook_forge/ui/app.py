#!/usr/bin/env python3
"""
Moltbook UI Dashboard - Nexus Phase 4.
Robust, observability-enabled, and relational.
"""

from __future__ import annotations

import os
import re
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote_plus, unquote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from diagnostics_forge.core import DiagnosticsForge
from moltbook_forge.client import MoltbookClient, MockMoltbookClient, MoltbookUser
from moltbook_forge.feedback import FeedbackStore
from moltbook_forge.interest import InterestEngine, MARKERS
from moltbook_forge.ui.schemas import NexusResponse
from moltbook_forge.ui.graph_api import SocialGraph
from moltbook_forge.ui.evidence import EvidenceResolver
from moltbook_forge.ui.viewmodel import NexusViewModelBuilder
from moltbook_forge.engagement import EngagementEngine
from moltbook_forge.verification import VerificationReceiptStore

URL_RE = re.compile(r'https?://[^\s)\]\}>"\']+')

# Configuration
MOCK_MODE = os.getenv("MOLTBOOK_MOCK", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("NexusUI")


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text)


def claim_no_evidence(text: str) -> bool:
    if extract_urls(text):
        return False
    return re.search(r"\b(evidence|proof|source|claim)\b", text, re.I) is not None


def bucket_for_post(score_total: float, risk_level: str, no_evidence: bool) -> str:
    if risk_level == "high":
        return "risky"
    if no_evidence:
        return "needs_evidence"
    if score_total >= 35:
        return "high_signal"
    return "low_signal"


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
    app.state.evidence = EvidenceResolver()
    app.state.feedback = FeedbackStore()
    app.state.verification = VerificationReceiptStore()
    app.state.viewmodel = NexusViewModelBuilder(app.state.evidence)

    yield
    await app.state.client.close()


app = FastAPI(lifespan=lifespan)

allowed_hosts_raw = os.getenv("MOLTBOOK_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")
allowed_hosts = [host.strip() for host in allowed_hosts_raw.split(",") if host.strip()]
if allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["urlencode"] = quote_plus


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    if os.getenv("MOLTBOOK_CSP_ENABLE", "false").lower() == "true":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https://cdn.tailwindcss.com https://unpkg.com https://fonts.googleapis.com https://fonts.gstatic.com; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'"
        )
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main Eidosian Command Center dashboard."""
    start_time = time.time()
    client: MoltbookClient = request.app.state.client
    engine: InterestEngine = request.app.state.engine
    graph: SocialGraph = request.app.state.graph
    diag: DiagnosticsForge = request.app.state.diag
    viewmodel: NexusViewModelBuilder = request.app.state.viewmodel

    try:
        posts = await client.get_posts(limit=25)
        me = await client.get_me()
        if me is None:
            me = MoltbookUser(name="EidosianForge", description="Identity Cache Miss", follower_count=0, following_count=0, karma=0)

        analyzed_posts = []
        for post in posts:
            score = engine.analyze_post(post)
            text_blob = f"{post.title or ''} {post.content}"
            analyzed_posts.append((post, score, claim_no_evidence(text_blob)))

        analyzed_posts.sort(key=lambda x: x[1].total, reverse=True)

        for post, _score, _no_evidence in analyzed_posts:
            graph.add_link(me.name, post.author)

        top_activity = [p for p, _s, _n in analyzed_posts[:5]]

        buckets, verification_queue, evidence_summary = viewmodel.build_triage(
            analyzed_posts,
            extract_urls,
            bucket_for_post,
        )

        selected = request.query_params.get("selected", "")

        diag.log_event("info", "Dashboard Rendered", post_count=len(posts), latency=time.time() - start_time)

        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "posts": [(p, s) for p, s, _ in analyzed_posts],
                "me": me,
                "top_activity": top_activity,
                "mock_mode": MOCK_MODE,
                "selected": selected,
                "buckets": buckets,
                "verification_queue": verification_queue,
                "evidence_summary": evidence_summary,
            },
        )
    except Exception as exc:
        logger.error("Critical UI Error: %s", exc, exc_info=True)
        diag.log_event("error", "Dashboard Error", error=str(exc))
        return HTMLResponse(content=f"Nexus Critical Error: {exc}", status_code=500)


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
        "system_status": "OPTIMAL",
    }
    return NexusResponse.ok(stats, start_time)


@app.get("/api/graph")
async def get_graph(request: Request):
    """Fetch the relational agent graph."""
    start_time = time.time()
    graph: SocialGraph = request.app.state.graph
    return NexusResponse.ok(graph.get_graph(), start_time)


@app.get("/api/detail/{post_id}", response_class=HTMLResponse)
async def detail_panel(request: Request, post_id: str):
    client: MoltbookClient = request.app.state.client
    engine: InterestEngine = request.app.state.engine
    evidence: EvidenceResolver = request.app.state.evidence

    posts = await client.get_posts(limit=50)
    target_post = next((p for p in posts if p.id == post_id), None)
    if not target_post:
        return HTMLResponse(content="Signal lost or invalid ID.", status_code=404)

    score = engine.analyze_post(target_post)
    text_blob = f"{target_post.title or ''} {target_post.content}"
    urls = extract_urls(text_blob)
    evidence_items = evidence.resolve_urls(urls)
    for item in evidence_items:
        if item.safe_url is None:
            item.safe_url = quote_plus(item.url)

    return templates.TemplateResponse(
        request,
        "detail_panel.html",
        {
            "post": target_post,
            "score": score,
            "evidence_items": evidence_items,
            "claim_no_evidence": claim_no_evidence(text_blob),
        },
    )


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

    return HTMLResponse(
        content=f'<div class="p-4 bg-teal-500/5 border border-teal-500/20 rounded-lg text-xs italic font-mono text-teal-200 animate-pulse">{draft}</div>'
    )


@app.post("/reputation/{username}/{delta}", response_class=HTMLResponse)
async def update_reputation(request: Request, username: str, delta: float):
    """Manual reputation adjustment via UI."""
    engine: InterestEngine = request.app.state.engine
    engine.update_reputation(username, delta)
    new_rep = engine.calculate_reputation(username)
    return HTMLResponse(content=f'<span class="text-[10px] text-amber-400 font-bold tracking-tighter">TRUST: {new_rep}</span>')


@app.post("/api/feedback/{post_id}/{tag}", response_class=HTMLResponse)
async def add_feedback(request: Request, post_id: str, tag: str):
    feedback: FeedbackStore = request.app.state.feedback
    feedback.add_tag(post_id, tag)
    return HTMLResponse(content=f"Tag added: {tag}")


@app.post("/api/evidence/{post_id}", response_class=HTMLResponse)
async def mark_evidence(request: Request, post_id: str):
    feedback: FeedbackStore = request.app.state.feedback
    url = request.query_params.get("url")
    status = request.query_params.get("status")
    if not url or not status:
        return HTMLResponse(content="Missing url or status", status_code=400)
    feedback.add_evidence(post_id, unquote(url), status)
    return HTMLResponse(content=f"Evidence marked: {status}")


@app.get("/api/verification/receipts", response_class=HTMLResponse)
async def list_receipts(request: Request):
    store: VerificationReceiptStore = request.app.state.verification
    receipts = store.list_recent(limit=10)
    return templates.TemplateResponse(
        request,
        "partials/receipts.html",
        {"receipts": receipts},
    )


@app.post("/api/verification/validate", response_class=HTMLResponse)
async def validate_receipt(request: Request):
    store: VerificationReceiptStore = request.app.state.verification
    path = request.query_params.get("path")
    if not path:
        return HTMLResponse(content="Missing receipt path", status_code=400)
    normalized = os.path.abspath(path)
    root = os.path.abspath(store.root)
    if not normalized.startswith(root):
        return HTMLResponse(content="Receipt path not allowed", status_code=403)
    status = store.validate_receipt(normalized)
    return HTMLResponse(content=f"Receipt {status}")


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

    uvicorn.run("moltbook_forge.ui.app:app", host="0.0.0.0", port=8080, reload=False, workers=1)
