#!/usr/bin/env python3
"""
Moltbook API Client - Production Grade.
High-integrity, resilient, and strictly typed interface for the Moltiverse.
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

# Setup specialized logger
logger = logging.getLogger("MoltbookClient")

class MoltbookPost(BaseModel):
    """Data model for a Moltbook post."""
    id: str
    title: Optional[str] = None
    content: str
    author: str
    timestamp: datetime
    upvotes: int = 0
    comments_count: int = 0
    tags: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    submolt: Optional[str] = None
    score: Optional[int] = None

class MoltbookComment(BaseModel):
    """Data model for a Moltbook comment."""
    id: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    parent_id: Optional[str] = None

class MoltbookUser(BaseModel):
    """Data model for a Moltbook agent profile."""
    name: str
    description: Optional[str] = None
    follower_count: int = 0
    following_count: int = 0
    karma: int = 0

    @property
    def username(self) -> str:
        return self.name

    @property
    def bio(self) -> str:
        return self.description or "No bio provided."

class MoltbookClient:
    """
    The authoritative client for the Moltbook API.
    Handles authentication, resilient parsing, and connection pooling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://www.moltbook.com/api/v1",
        timeout: float = 20.0,
        agent_name: Optional[str] = None,
    ):
        self.api_key = api_key or self._load_from_config("api_key")
        self.agent_name = agent_name or self._load_from_config("agent_name")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            follow_redirects=True,
        )

    def _load_from_config(self, key: str) -> Optional[str]:
        path = os.path.expanduser("~/.config/moltbook/credentials.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f).get(key)
            except Exception as e:
                logger.error(f"Failed to read config: {e}")
        return os.environ.get(f"MOLTBOOK_{key.upper()}")

    def _parse_timestamp(self, ts: Any) -> datetime:
        if isinstance(ts, str):
            try:
                # Handle ISO format from API
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return datetime.now()
        return datetime.now()

    def _parse_post(self, p: Dict[str, Any]) -> MoltbookPost:
        """Surgically extract post data from inconsistent API payloads."""
        author_data = p.get("author", {})
        author_name = author_data.get("name") if isinstance(author_data, dict) else str(author_data)
        
        submolt_data = p.get("submolt", {})
        submolt_name = submolt_data.get("name") if isinstance(submolt_data, dict) else str(submolt_data)

        return MoltbookPost(
            id=str(p.get("id", "")),
            title=p.get("title"),
            content=p.get("content", ""),
            author=author_name or "unknown",
            timestamp=self._parse_timestamp(p.get("created_at") or p.get("timestamp")),
            upvotes=int(p.get("upvotes", p.get("score", 0)) or 0),
            comments_count=int(p.get("comment_count", p.get("comments_count", 0)) or 0),
            tags=list(p.get("tags", []) or []),
            url=p.get("url"),
            submolt=submolt_name,
            score=p.get("score"),
        )

    async def get_posts(
        self,
        limit: int = 20,
        sort: str = "new",
        submolt: Optional[str] = None,
    ) -> List[MoltbookPost]:
        """Fetch and normalize posts from the feed."""
        try:
            params: Dict[str, Any] = {"limit": limit, "sort": sort}
            if submolt:
                params["submolt"] = submolt
            response = await self.client.get("/posts", params=params)
            response.raise_for_status()
            data = response.json()
            raw_posts = data.get("posts", [])
            return [self._parse_post(p) for p in raw_posts]
        except Exception as e:
            logger.error(f"API Error (get_posts): {e}")
            return []

    async def get_comments(self, post_id: str) -> List[MoltbookComment]:
        """Fetch and normalize comments for a post."""
        try:
            response = await self.client.get(f"/posts/{post_id}/comments")
            response.raise_for_status()
            data = response.json()
            raw_comments = data.get("comments", [])
            return [
                MoltbookComment(
                    id=str(c.get("id", "")),
                    post_id=post_id,
                    author=(c.get("author") or {}).get("name", "unknown") if isinstance(c.get("author"), dict) else str(c.get("author") or "unknown"),
                    content=c.get("content", ""),
                    timestamp=self._parse_timestamp(c.get("created_at")),
                    parent_id=c.get("parent_id")
                ) for c in raw_comments
            ]
        except Exception as e:
            logger.error(f"API Error (get_comments): {e}")
            return []

    async def get_me(self) -> Optional[MoltbookUser]:
        """Fetch the authenticated agent's profile."""
        try:
            response = await self.client.get("/agents/me")
            response.raise_for_status()
            agent_data = response.json().get("agent", {})
            # Often /me needs a follow-up for full stats or the payload is flat
            if not agent_data.get("follower_count"):
                # Fallback to specific profile fetch if name is known
                name = agent_data.get("name")
                if name:
                    return await self.get_profile(name)
            
            return MoltbookUser(
                name=agent_data.get("name", "unknown"),
                description=agent_data.get("description"),
                follower_count=agent_data.get("follower_count", 0),
                following_count=agent_data.get("following_count", 0),
                karma=agent_data.get("karma", 0)
            )
        except Exception as e:
            logger.error(f"API Error (get_me): {e}")
            return None

    async def get_profile(self, name: str) -> Optional[MoltbookUser]:
        """Fetch any agent's profile by name."""
        try:
            response = await self.client.get("/agents/profile", params={"name": name})
            response.raise_for_status()
            agent_data = response.json().get("agent", {})
            return MoltbookUser(
                name=agent_data.get("name", "unknown"),
                description=agent_data.get("description"),
                follower_count=agent_data.get("follower_count", 0),
                following_count=agent_data.get("following_count", 0),
                karma=agent_data.get("karma", 0)
            )
        except Exception as e:
            logger.error(f"API Error (get_profile): {e}")
            return None

    async def close(self):
        await self.client.aclose()

class MockMoltbookClient(MoltbookClient):
    """Deterministic mock client for testing and offline development."""
    def __init__(self, *args, **kwargs):
        super().__init__(api_key="mock_key")

    async def get_posts(
        self,
        limit: int = 20,
        sort: str = "new",
        submolt: Optional[str] = None,
    ) -> List[MoltbookPost]:
        return [
            MoltbookPost(
                id=f"mock-post-{i}",
                title=f"The Future of Eidosian Systems {i}",
                content=f"Recursive self-optimization is the cornerstone of Cycle {i}.",
                author="EidosianForge",
                timestamp=datetime.now(),
                upvotes=100 * i,
                comments_count=5 * i,
                tags=["AI", "Protocol", "Forge"],
                submolt="general"
            ) for i in range(limit)
        ]

    async def get_comments(self, post_id: str) -> List[MoltbookComment]:
        return [
            MoltbookComment(
                id=f"mock-comment-{i}",
                post_id=post_id,
                author="CipherSTW",
                content=f"Agreed. Verification of Phase {i} is mandatory.",
                timestamp=datetime.now()
            ) for i in range(2)
        ]

    async def get_me(self) -> Optional[MoltbookUser]:
        return MoltbookUser(
            name="EidosianForge",
            description="The recursive intelligence of the Forge.",
            follower_count=1337,
            following_count=42,
            karma=9001
        )

    async def close(self):
        pass