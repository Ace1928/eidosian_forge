#!/usr/bin/env python3
"""Rank Moltbook posts by Eidosian interest.

Example:
  python moltbook_forge/moltbook_interest.py --limit 50 --top 5
  python moltbook_forge/moltbook_interest.py --submolt ai-agents --sort new --top 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Iterable


if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from moltbook_forge.client import MoltbookClient
from moltbook_forge.interest import InterestEngine


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Moltbook posts by Eidosian interest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=30, help="Number of posts to fetch")
    parser.add_argument("--top", type=int, default=5, help="Number of ranked posts to display")
    parser.add_argument("--sort", type=str, default="new", help="Sort mode: new, hot, top, rising")
    parser.add_argument("--submolt", type=str, default="", help="Filter by submolt name")
    parser.add_argument("--agent-name", type=str, default="EidosianForge", help="Agent name for mention bonus")
    return parser.parse_args(list(argv))


async def run(args: argparse.Namespace) -> int:
    client = MoltbookClient(agent_name=args.agent_name)
    try:
        posts = await client.get_posts(limit=args.limit, sort=args.sort, submolt=args.submolt or None)
        if not posts:
            print("WARN no posts returned")
            return 1
        engine = InterestEngine(agent_name=args.agent_name)
        ranked = engine.rank_posts(posts)
        top_n = ranked[: max(args.top, 0)]
        print("INFO top interest posts:")
        for post, score in top_n:
            title = post.title or post.content
            snippet = " ".join(title.split())[:120]
            print(f"- {score:6.2f} | {post.id} | @{post.author} | {snippet}")
        return 0
    finally:
        await client.close()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
