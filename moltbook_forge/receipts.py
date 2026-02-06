#!/usr/bin/env python3
"""Proof-of-process receipts for Moltbook/Eidosian workflows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[str] = None
    duration_ms: Optional[float] = None
    outcome: Optional[str] = None


class Artifact(BaseModel):
    path: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None


class Receipt(BaseModel):
    receipt_id: str
    created_at: str
    input_hash: str
    input_path: Optional[str] = None
    summary: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    output_hash: Optional[str] = None
    signature: Optional[str] = None

    def content_hash(self) -> str:
        payload = self.model_dump(exclude={"signature"}, by_alias=True)
        return _sha256(json.dumps(payload, sort_keys=True))

    def sign(self) -> None:
        self.signature = self.content_hash()

    def verify(self) -> bool:
        if not self.signature:
            return False
        return self.signature == self.content_hash()


def build_receipt(
    input_text: str,
    input_path: Optional[str] = None,
    summary: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None,
    artifacts: Optional[List[Artifact]] = None,
    output_text: Optional[str] = None,
) -> Receipt:
    created_at = datetime.now(timezone.utc).isoformat()
    receipt = Receipt(
        receipt_id=_sha256(f"{created_at}:{input_text}")[:16],
        created_at=created_at,
        input_hash=_sha256(input_text),
        input_path=input_path,
        summary=summary,
        tool_calls=tool_calls or [],
        artifacts=artifacts or [],
        output_hash=_sha256(output_text) if output_text else None,
    )
    return receipt


def load_receipt(path: str) -> Receipt:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Receipt.model_validate(payload)


def save_receipt(path: str, receipt: Receipt) -> None:
    if receipt.signature is None:
        receipt.sign()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(receipt.model_dump(by_alias=True), handle, indent=2, sort_keys=True)
