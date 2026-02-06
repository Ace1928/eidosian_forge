#!/usr/bin/env python3
"""Live Moltbook scan utility with scored output."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from moltbook_forge.client import MoltbookClient, MockMoltbookClient, MoltbookPost
from moltbook_forge.interest import InterestEngine


@dataclass
class ScanSummary:
    total: int
    top_posts: list[dict]
    submolts: list[tuple[str, int]]
    intents: list[tuple[str, int]]
    risks: list[tuple[str, int]]
    keywords: list[tuple[str, int]]


def summarize_posts(posts: list[dict], top_n: int = 10) -> ScanSummary:
    posts_sorted = sorted(posts, key=lambda x: x.get("score", 0), reverse=True)
    top_posts = []
    for p in posts_sorted[:top_n]:
        title = (p.get("title") or p.get("content") or "").replace("\n", " ")
        title = " ".join(title.split())
        top_posts.append(
            {
                "id": p.get("id"),
                "author": p.get("author"),
                "submolt": p.get("submolt") or "unknown",
                "score": p.get("score"),
                "risk": p.get("risk"),
                "intent": p.get("intent"),
                "snippet": title[:160],
            }
        )
    submolts = Counter(p.get("submolt") or "unknown" for p in posts)
    intents = Counter(p.get("intent") or "unknown" for p in posts)
    risks = Counter(p.get("risk") or "unknown" for p in posts)
    keywords = Counter()
    for p in posts:
        for kw in p.get("keywords") or []:
            keywords[kw.lower()] += 1
    return ScanSummary(
        total=len(posts),
        top_posts=top_posts,
        submolts=submolts.most_common(),
        intents=intents.most_common(),
        risks=risks.most_common(),
        keywords=keywords.most_common(15),
    )


async def fetch_posts(limit: int, sort: str, mock: bool) -> list[MoltbookPost]:
    client = MockMoltbookClient() if mock else MoltbookClient()
    try:
        return await client.get_posts(limit=limit, sort=sort)
    finally:
        await client.close()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Moltbook feed and persist scored output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=60, help="Posts to fetch")
    parser.add_argument("--sort", type=str, default="new", help="Sort mode")
    parser.add_argument("--top", type=int, default=10, help="Top N in summary")
    parser.add_argument("--mock", action="store_true", help="Use mock client")
    return parser.parse_args(list(argv))


def write_summary(path: Path, summary: ScanSummary, source_path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Moltbook Live Scan Summary\n\n")
        handle.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        handle.write(f"Source: {source_path}\n\n")
        handle.write("## Top Signal Posts\n")
        for item in summary.top_posts:
            handle.write(
                f"- {item['score']:6.2f} | {item['risk']} | {item['intent']} | "
                f"@{item['author']} | {item['submolt']} | {item['id']} | {item['snippet']}\n"
            )
        handle.write("\n## Submolt Distribution\n")
        for name, count in summary.submolts:
            handle.write(f"- {name}: {count}\n")
        handle.write("\n## Intent Distribution\n")
        for name, count in summary.intents:
            handle.write(f"- {name}: {count}\n")
        handle.write("\n## Risk Distribution\n")
        for name, count in summary.risks:
            handle.write(f"- {name}: {count}\n")
        handle.write("\n## Keyword Frequency (top 15)\n")
        for kw, count in summary.keywords:
            handle.write(f"- {kw}: {count}\n")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    posts = asyncio.run(fetch_posts(args.limit, args.sort, args.mock))
    engine = InterestEngine()
    scored = []
    for p in posts:
        breakdown = engine.analyze_post(p)
        scored.append(
            {
                "id": p.id,
                "author": p.author,
                "title": p.title,
                "content": p.content,
                "submolt": p.submolt,
                "timestamp": p.timestamp.isoformat(),
                "upvotes": p.upvotes,
                "comments": p.comments_count,
                "score": breakdown.total,
                "intent": breakdown.llm_intent,
                "risk": breakdown.risk_level,
                "keywords": breakdown.matched_keywords,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)

    out_dir = Path("data/moltbook")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"live_scan_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "count": len(scored),
                "posts": scored,
            },
            handle,
            indent=2,
        )

    summary = summarize_posts(scored, top_n=args.top)
    summary_path = out_dir / f"live_scan_{stamp}.md"
    write_summary(summary_path, summary, out_path)

    print(out_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
