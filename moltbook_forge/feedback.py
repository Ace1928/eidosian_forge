#!/usr/bin/env python3
"""Feedback store for Moltbook Nexus triage actions."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List


class FeedbackStore:
    """Simple JSON-backed feedback store for post tags and notes."""

    def __init__(self, path: str = "moltbook_forge/data/feedback.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data: Dict[str, Dict[str, List[str] | Dict[str, Dict[str, str]]]] = self._load()

    def _load(self) -> Dict[str, Dict[str, List[str]]]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)

    def add_tag(self, post_id: str, tag: str) -> None:
        entry = self._data.setdefault(post_id, {"tags": [], "history": [], "evidence": {}})
        if tag not in entry["tags"]:
            entry["tags"].append(tag)
        entry["history"].append({"tag": tag, "ts": datetime.now(timezone.utc).isoformat()})
        self._save()

    def add_evidence(self, post_id: str, url: str, status: str) -> None:
        entry = self._data.setdefault(post_id, {"tags": [], "history": [], "evidence": {}})
        evidence = entry.setdefault("evidence", {})
        if url not in evidence:
            evidence[url] = {"status": status, "history": []}
        evidence[url]["status"] = status
        evidence[url]["history"].append({"status": status, "ts": datetime.now(timezone.utc).isoformat()})
        self._save()

    def get_tags(self, post_id: str) -> List[str]:
        return self._data.get(post_id, {}).get("tags", [])

    def get_all_tags(self) -> Dict[str, List[str]]:
        return {pid: v.get("tags", []) for pid, v in self._data.items()}

    def get_evidence(self, post_id: str) -> Dict[str, Dict[str, str]]:
        entry = self._data.get(post_id, {})
        evidence = entry.get("evidence", {})
        if isinstance(evidence, dict):
            return evidence
        return {}
