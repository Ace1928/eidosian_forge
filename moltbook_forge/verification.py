#!/usr/bin/env python3
"""Verification receipt utilities for Moltbook Nexus."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from moltbook_forge.receipts import Artifact, ToolCall, build_receipt, load_receipt, save_receipt


@dataclass
class VerificationPayload:
    post_id: str
    author: str
    title: str | None
    content: str
    urls: List[str]
    evidence_status: Dict[str, Dict[str, str]]


class VerificationReceiptStore:
    def __init__(self, root: str = "moltbook_forge/data/verification_receipts") -> None:
        self.root = os.getenv("MOLTBOOK_VERIFICATION_ROOT", root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, post_id: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.root, f"{post_id}_{stamp}.json")

    def list_recent(self, limit: int = 20) -> List[str]:
        entries = []
        for name in os.listdir(self.root):
            if not name.endswith(".json"):
                continue
            entries.append(os.path.join(self.root, name))
        entries.sort(reverse=True)
        return entries[:limit]

    def validate_receipt(self, path: str) -> str:
        receipt = load_receipt(path)
        return "valid" if receipt.verify() else "invalid"

    def create_receipt(self, payload: VerificationPayload, summary: str) -> str:
        input_text = json.dumps(
            {
                "post_id": payload.post_id,
                "author": payload.author,
                "title": payload.title,
                "content": payload.content,
                "urls": payload.urls,
                "evidence_status": payload.evidence_status,
            },
            sort_keys=True,
        )
        tool_calls = [
            ToolCall(
                tool="verification.start",
                args={"post_id": payload.post_id, "url_count": len(payload.urls)},
                started_at=datetime.now(timezone.utc).isoformat(),
                outcome="started",
            )
        ]
        receipt = build_receipt(
            input_text=input_text,
            summary=summary,
            tool_calls=tool_calls,
            artifacts=[Artifact(path=f"moltbook:{payload.post_id}")],
            output_text="verification_started",
        )
        path = self._path(payload.post_id)
        save_receipt(path, receipt)
        return path
