#!/usr/bin/env python3
"""
Evidence resolution utilities for Moltbook Nexus.
Optionally fetch metadata for URLs if enabled via env var.
"""

from __future__ import annotations

import json
import os
import re
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import urlparse, quote

try:
    import httpx
    HAS_HTTPX = True
except Exception:
    HAS_HTTPX = False

TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

@dataclass
class EvidenceItem:
    url: str
    domain: str
    title: str | None = None
    fetched_at: str | None = None
    status: str | None = None
    safe_url: str | None = None
    credibility_score: int | None = None
    credibility_label: str | None = None


class CredibilityProvider:
    """Interface for credibility scoring providers."""

    def score(self, domain: str) -> tuple[int, str]:
        raise NotImplementedError


class DomainListCredibilityProvider(CredibilityProvider):
    def __init__(self, trusted: set[str], risky: set[str], unknown_score: int) -> None:
        self.trusted = trusted
        self.risky = risky
        self.unknown_score = unknown_score

    def score(self, domain: str) -> tuple[int, str]:
        if domain in self.trusted:
            score = 90
        elif domain in self.risky:
            score = 20
        else:
            score = self.unknown_score
        if score >= 80:
            label = "high"
        elif score >= 50:
            label = "medium"
        else:
            label = "low"
        return score, label


class EvidenceResolver:
    """Resolves URL metadata with an on-disk cache."""

    def __init__(self, cache_path: str = "moltbook_forge/data/evidence_cache.json") -> None:
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._cache: Dict[str, EvidenceItem] = self._load()
        self._async_enabled = os.getenv("MOLTBOOK_EVIDENCE_ASYNC", "false").lower() == "true"
        self._queue: asyncio.Queue[str] | None = None
        self._trusted_domains = self._load_domain_set("MOLTBOOK_EVIDENCE_TRUSTED_DOMAINS", [
            "github.com",
            "gitlab.com",
            "bitbucket.org",
            "arxiv.org",
            "doi.org",
            "openai.com",
            "pytorch.org",
            "numpy.org",
            "python.org",
            "ietf.org",
            "w3.org",
            "owasp.org",
        ])
        self._risky_domains = self._load_domain_set("MOLTBOOK_EVIDENCE_RISKY_DOMAINS", [
            "bit.ly",
            "tinyurl.com",
            "t.co",
            "is.gd",
            "goo.gl",
            "linktr.ee",
        ])
        self._unknown_score = int(os.getenv("MOLTBOOK_EVIDENCE_UNKNOWN_SCORE", "50"))
        self._providers: list[CredibilityProvider] = [
            DomainListCredibilityProvider(self._trusted_domains, self._risky_domains, self._unknown_score)
        ]

    def _load(self) -> Dict[str, EvidenceItem]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                return {url: EvidenceItem(**item) for url, item in raw.items()}
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        serialized = {url: item.__dict__ for url, item in self._cache.items()}
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(serialized, handle, indent=2, sort_keys=True)

    def _load_domain_set(self, env_key: str, defaults: List[str]) -> set[str]:
        raw = os.getenv(env_key, "")
        domains = {d.strip().lower() for d in raw.split(",") if d.strip()}
        if not domains:
            domains = {d.lower() for d in defaults}
        return domains

    def _domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc or "unknown"
        except Exception:
            return "unknown"

    def _safe_url(self, url: str) -> str:
        return quote(url, safe="")

    def _normalize_domain(self, domain: str) -> str:
        cleaned = domain.lower().strip()
        if cleaned.startswith("www."):
            cleaned = cleaned[4:]
        return cleaned

    def register_provider(self, provider: CredibilityProvider) -> None:
        self._providers.append(provider)

    def score_domain(self, domain: str) -> tuple[int, str]:
        normalized = self._normalize_domain(domain)
        for provider in self._providers:
            score, label = provider.score(normalized)
            if score is not None:
                return score, label
        return self._unknown_score, "medium"

    def _fetch_title(self, url: str) -> str | None:
        if not HAS_HTTPX:
            return None
        timeout = float(os.getenv("MOLTBOOK_EVIDENCE_TIMEOUT", "6"))
        headers = {"User-Agent": "MoltbookNexusEvidence/1.0"}
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
                resp = client.get(url)
                if resp.status_code >= 400:
                    return None
                match = TITLE_RE.search(resp.text[:200000])
                if match:
                    title = match.group(1)
                    return re.sub(r"\s+", " ", title).strip()
        except Exception:
            return None
        return None

    async def _fetch_title_async(self, url: str) -> str | None:
        if not HAS_HTTPX:
            return None
        timeout = float(os.getenv("MOLTBOOK_EVIDENCE_TIMEOUT", "6"))
        headers = {"User-Agent": "MoltbookNexusEvidence/1.0"}
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
                resp = await client.get(url)
                if resp.status_code >= 400:
                    return None
                match = TITLE_RE.search(resp.text[:200000])
                if match:
                    title = match.group(1)
                    return re.sub(r"\s+", " ", title).strip()
        except Exception:
            return None
        return None

    def enable_async_fetch(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()

    async def enqueue_fetch(self, url: str) -> None:
        if not self._async_enabled:
            return
        if self._queue is None:
            self.enable_async_fetch()
        if self._queue:
            await self._queue.put(url)

    async def run_fetch_worker(self) -> None:
        if not self._async_enabled:
            return
        if self._queue is None:
            self.enable_async_fetch()
        if self._queue is None:
            return
        while True:
            url = await self._queue.get()
            item = self._cache.get(url)
            if item and item.title:
                self._queue.task_done()
                continue
            title = await self._fetch_title_async(url)
            if title:
                if not item:
                    item = EvidenceItem(url=url, domain=self._domain(url))
                    score, label = self.score_domain(item.domain)
                    item.credibility_score = score
                    item.credibility_label = label
                if not item.safe_url:
                    item.safe_url = self._safe_url(url)
                item.title = title
                item.fetched_at = datetime.now(timezone.utc).isoformat()
                self._cache[url] = item
                self._save()
            self._queue.task_done()

    def resolve_urls(self, urls: List[str]) -> List[EvidenceItem]:
        fetch_enabled = os.getenv("MOLTBOOK_EVIDENCE_FETCH", "false").lower() == "true"
        resolved: List[EvidenceItem] = []
        updated = False
        for url in urls:
            if url in self._cache:
                cached = self._cache[url]
                if cached.credibility_score is None or cached.credibility_label is None:
                    score, label = self.score_domain(cached.domain)
                    cached.credibility_score = score
                    cached.credibility_label = label
                    updated = True
                if not cached.safe_url:
                    cached.safe_url = self._safe_url(cached.url)
                resolved.append(cached)
                continue
            item = EvidenceItem(url=url, domain=self._domain(url))
            item.safe_url = self._safe_url(url)
            score, label = self.score_domain(item.domain)
            item.credibility_score = score
            item.credibility_label = label
            if fetch_enabled and not self._async_enabled:
                title = self._fetch_title(url)
                if title:
                    item.title = title
                    item.fetched_at = datetime.now(timezone.utc).isoformat()
            elif fetch_enabled and self._async_enabled:
                # enqueue async fetch and return placeholder
                if self._queue is not None:
                    try:
                        self._queue.put_nowait(url)
                    except Exception:
                        pass
            self._cache[url] = item
            resolved.append(item)
            updated = True
        if updated:
            self._save()
        return resolved
