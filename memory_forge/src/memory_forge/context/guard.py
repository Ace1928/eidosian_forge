#!/usr/bin/env python3
"""Context redaction and retention enforcement utilities."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

REDACTION_RULES = [
    ("moltbook_api_key", re.compile(r"\bmoltbook_sk_[A-Za-z0-9]{6,}\b")),
    ("moltdev_key", re.compile(r"\bmoltdev_[A-Za-z0-9]{6,}\b")),
    ("generic_api_key", re.compile(r"\bapi[_-]?key\b[:=]\s*[^\s]+", re.I)),
    ("auth_token", re.compile(r"\bauth(?:orization)?[_-]?token\b[:=]\s*[^\s]+", re.I)),
    ("ip_address", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    ("email_address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


@dataclass
class RedactionFinding:
    label: str
    count: int


@dataclass
class ContextSBOM:
    created_at: str
    sources: List[str]
    redactions: List[RedactionFinding]
    outputs: List[str]
    retention_days: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "sources": self.sources,
            "redactions": [f.__dict__ for f in self.redactions],
            "outputs": self.outputs,
            "retention_days": self.retention_days,
        }


def redact_sensitive(text: str) -> tuple[str, List[RedactionFinding]]:
    redacted = text
    findings: List[RedactionFinding] = []
    for label, pattern in REDACTION_RULES:
        matches = pattern.findall(redacted)
        if matches:
            redacted = pattern.sub(f"[REDACTED:{label.upper()}]", redacted)
            findings.append(RedactionFinding(label=label, count=len(matches)))
    return redacted, findings


def write_sbom(path: Path, sbom: ContextSBOM) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sbom.to_dict(), indent=2), encoding="utf-8")


def retention_cleanup(target_dir: Path, retention_days: int, dry_run: bool = False) -> List[str]:
    if retention_days <= 0:
        return []
    cutoff = time.time() - (retention_days * 86400)
    removed: List[str] = []
    for root, _, files in os.walk(target_dir):
        for name in files:
            path = Path(root) / name
            try:
                if path.stat().st_mtime < cutoff:
                    removed.append(str(path))
                    if not dry_run:
                        path.unlink()
            except FileNotFoundError:
                continue
    return removed


def build_sbom(
    sources: List[str],
    redactions: List[RedactionFinding],
    outputs: List[str],
    retention_days: Optional[int] = None,
) -> ContextSBOM:
    return ContextSBOM(
        created_at=datetime.now(timezone.utc).isoformat(),
        sources=sources,
        redactions=redactions,
        outputs=outputs,
        retention_days=retention_days,
    )
