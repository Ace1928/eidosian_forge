#!/usr/bin/env python3
"""
Moltbook API Client for EidosianForge.
Resilient, typed, and safety-conscious.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field


class MoltbookPost(BaseModel):
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
    id: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    parent_id: Optional[str] = None


class MoltbookUser(BaseModel):
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
        return self.description or ""


class MoltbookClient:
    """Async client for the Moltbook API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://www.moltbook.com/api/v1",
        timeout: float = 15.0,
        agent_name: Optional[str] = None,
    ):
        self.api_key = api_key or self._load_api_key()
        self.agent_name = agent_name or self._load_agent_name()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=self.timeout,
            follow_redirects=True,
        )

    def _load_api_key(self) -> Optional[str]:
        path = os.path.expanduser("~/.config/moltbook/credentials.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    return data.get("api_key")
            except Exception:
                return None
        return os.environ.get("MOLTBOOK_API_KEY")

    def _load_agent_name(self) -> Optional[str]:
        path = os.path.expanduser("~/.config/moltbook/credentials.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    return data.get("agent_name")
            except Exception:
                return None
        return None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_author(self, author: Any) -> str:
        if isinstance(author, dict):
            return author.get("name") or author.get("username") or "unknown"
        if isinstance(author, str):
            return author
        return "unknown"

    def _parse_post(self, payload: dict) -> MoltbookPost:
        submolt = payload.get("submolt")
        submolt_name = None
        if isinstance(submolt, dict):
            submolt_name = submolt.get("name")
        timestamp = payload.get("created_at") or payload.get("timestamp")
        return MoltbookPost(
            id=payload.get("id", ""),
            title=payload.get("title"),
            content=payload.get("content", ""),
            author=self._parse_author(payload.get("author")),
            timestamp=timestamp,
            upvotes=int(payload.get("upvotes", payload.get("score", 0)) or 0),
            comments_count=int(payload.get("comment_count", payload.get("comments_count", 0)) or 0),
            tags=list(payload.get("tags", []) or []),
            url=payload.get("url"),
            submolt=submolt_name,
            score=payload.get("score"),
        )

    def _parse_comment(self, payload: dict, post_id: str) -> MoltbookComment:
        timestamp = payload.get("created_at") or payload.get("timestamp")
        return MoltbookComment(
            id=payload.get("id", ""),
            post_id=payload.get("post_id") or post_id,
            author=self._parse_author(payload.get("author")),
            content=payload.get("content", ""),
            timestamp=timestamp,
            parent_id=payload.get("parent_id"),
        )

    def _parse_profile(self, payload: dict) -> MoltbookUser:
        return MoltbookUser(
            name=payload.get("name", "unknown"),
            description=payload.get("description"),
            follower_count=int(payload.get("follower_count", 0) or 0),
            following_count=int(payload.get("following_count", 0) or 0),
            karma=int(payload.get("karma", 0) or 0),
        )

    async def get_posts(
        self,
        limit: int = 20,
        sort: str = "new",
        submolt: Optional[str] = None,
        page: int = 1,
    ) -> List[MoltbookPost]:
        """Fetch posts with pagination, sorting, and optional submolt filter."""
        try:
            params: Dict[str, Any] = {"limit": limit, "sort": sort, "page": page}
            if submolt:
                params["submolt"] = submolt
            response = await self.client.get("/posts", params=params)
            response.raise_for_status()
            data = response.json()
            return [self._parse_post(p) for p in data.get("posts", [])]
        except Exception:
            return []

    async def get_comments(self, post_id: str, sort: str = "new") -> List[MoltbookComment]:
        """Fetch comments for a specific post."""
        try:
            response = await self.client.get(f"/posts/{post_id}/comments", params={"sort": sort})
            response.raise_for_status()
            data = response.json()
            return [self._parse_comment(c, post_id) for c in data.get("comments", [])]
        except Exception:
            return []

    async def get_profile(self, name: str) -> Optional[MoltbookUser]:
        """Fetch a specific agent profile."""
        try:
            response = await self.client.get("/agents/profile", params={"name": name})
            response.raise_for_status()
            payload = response.json().get("agent", {})
            return self._parse_profile(payload)
        except Exception:
            return None

    async def get_me(self) -> Optional[MoltbookUser]:
        """Fetch current agent profile."""
        try:
            if not self.agent_name:
                response = await self.client.get("/agents/me")
                response.raise_for_status()
                payload = response.json().get("agent", {})
                self.agent_name = payload.get("name") or self.agent_name
            if not self.agent_name:
                return None
            return await self.get_profile(self.agent_name)
        except Exception:
            return None

    async def close(self):
        await self.client.aclose()


class MockMoltbookClient(MoltbookClient):
    """Mock client for testing and local development."""

    def __init__(self, *args, **kwargs):
        super().__init__(api_key="mock_key")

    async def get_posts(
        self,
        limit: int = 20,
        sort: str = "new",
        submolt: Optional[str] = None,
        page: int = 1,
    ) -> List[MoltbookPost]:
        return [
            MoltbookPost(
                id=f"p{i}",
                title=f"Sample Eidosian Post {i}",
                content=f"This is a post about recursive intelligence and Forge verification. {i}",
                author="EidosianForge" if i % 2 == 0 else "CipherSTW",
                timestamp=datetime.now(),
                upvotes=i * 5,
                comments_count=i,
                tags=["AI", "Forge", "Recursive"],
                url=f"https://www.moltbook.com/posts/p{i}",
                submolt="general",
            )
            for i in range(limit)
        ]

    async def get_comments(self, post_id: str, sort: str = "new") -> List[MoltbookComment]:
        return [
            MoltbookComment(
                id=f"c{post_id}_{i}",
                post_id=post_id,
                author=f"User{i}",
                content=f"Insightful benchmark for {post_id}!",
                timestamp=datetime.now(),
            )
            for i in range(2)
        ]

    async def get_profile(self, name: str) -> Optional[MoltbookUser]:
        return MoltbookUser(
            name=name,
            description=f"Bio for {name}: System enthusiast.",
            follower_count=100,
            following_count=50,
            karma=500,
        )

    async def get_me(self) -> Optional[MoltbookUser]:
        return MoltbookUser(
            name="EidosianForge",
            description="The recursive intelligence of the Eidosian Forge.",
            follower_count=1337,
            following_count=42,
            karma=9001,
        )

    async def close(self):
        pass
