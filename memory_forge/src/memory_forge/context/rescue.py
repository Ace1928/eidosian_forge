#!/usr/bin/env python3
"""Memory compression rescue kit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class RescueTemplate:
    path: Path
    content: str


def precompression_prompt() -> str:
    return (
        "Rescue Summary (pre-compression):\n"
        "- Current objective:\n"
        "- Key decisions made:\n"
        "- Open questions / blockers:\n"
        "- Next concrete steps:\n"
        "- Files touched:\n"
    )


def build_templates(root: Path) -> List[RescueTemplate]:
    today = datetime.now().strftime("%Y-%m-%d")
    memory_dir = root / "memory"
    templates = [
        RescueTemplate(
            memory_dir / "README.md",
            "# Memory Rescue Kit\n\n"
            "- `index.json` contains gist entries for fast recall.\n"
            "- `last_session.md` is the rescue summary before compression.\n"
            "- `daily/` holds raw daily logs.\n"
            "- `weekly/` holds distilled weekly summaries.\n",
        ),
        RescueTemplate(
            memory_dir / "index.json",
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"entries\": []\n"
            "}\n",
        ),
        RescueTemplate(
            memory_dir / "last_session.md",
            precompression_prompt(),
        ),
        RescueTemplate(
            memory_dir / "daily" / f"{today}.md",
            f"# Daily Log {today}\n\n- Decisions:\n- Progress:\n- Blockers:\n- Next:\n",
        ),
        RescueTemplate(
            memory_dir / "weekly" / "README.md",
            "# Weekly Summaries\n\n- One summary per week.\n",
        ),
    ]
    return templates


def generate_rescue_kit(root: Path) -> List[Path]:
    root = root.expanduser()
    created: List[Path] = []
    for template in build_templates(root):
        template.path.parent.mkdir(parents=True, exist_ok=True)
        if not template.path.exists():
            template.path.write_text(template.content, encoding="utf-8")
            created.append(template.path)
    return created
