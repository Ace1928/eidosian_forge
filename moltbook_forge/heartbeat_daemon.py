#!/usr/bin/env python3
"""
Moltbook Heartbeat Daemon.
Monitors the feed and optionally stores high-signal content in MemoryForge.

Usage:
  python moltbook_forge/heartbeat_daemon.py --allow-network
  python moltbook_forge/heartbeat_daemon.py --allow-network --interval 600 --threshold 40 --once
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

from moltbook_forge.client import MoltbookClient
from moltbook_forge.interest import InterestEngine

try:
    from memory_forge.core.main import MemoryForge
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False


def build_logger(log_file: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("NexusHeartbeat")


class HeartbeatDaemon:
    def __init__(
        self,
        interval: int = 300,
        threshold: float = 30.0,
        limit: int = 20,
        max_seen: int = 1000,
        use_memory: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.interval = interval
        self.threshold = threshold
        self.limit = limit
        self.max_seen = max_seen
        self.client = MoltbookClient()
        self.engine = InterestEngine()
        self.logger = logger or logging.getLogger("NexusHeartbeat")
        self.memory = MemoryForge() if HAS_MEMORY and use_memory else None
        self.seen_posts: set[str] = set()
        self.is_running = False

    async def cycle(self):
        """A single monitoring cycle."""
        self.logger.info("Starting signal scan")
        posts = await self.client.get_posts(limit=self.limit)

        high_signal_count = 0
        for post in posts:
            if post.id in self.seen_posts:
                continue

            self.seen_posts.add(post.id)
            analysis = self.engine.analyze_post(post)

            if analysis.total >= self.threshold:
                self.logger.info(
                    "HIGH SIGNAL: @%s | Score: %.2f | Intent: %s",
                    post.author,
                    analysis.total,
                    analysis.llm_intent,
                )
                high_signal_count += 1

                if self.memory:
                    memory_content = (
                        f"Moltbook Post by @{post.author} at {post.timestamp}\n"
                        f"Intent: {analysis.llm_intent}\n"
                        f"Content: {post.content}\n"
                        f"Keywords: {', '.join(analysis.matched_keywords)}"
                    )
                    try:
                        mem_id = self.memory.remember(
                            content=memory_content,
                            metadata={
                                "source": "moltbook",
                                "author": post.author,
                                "intent": analysis.llm_intent,
                                "post_id": post.id,
                                "score": analysis.total,
                            },
                        )
                        self.logger.info("Stored in memory: %s", mem_id)
                    except Exception as exc:
                        self.logger.error("Failed to store memory: %s", exc)

        self.logger.info("Cycle complete. New high-signal items: %d", high_signal_count)
        if len(self.seen_posts) > self.max_seen:
            self.seen_posts = set(list(self.seen_posts)[-self.max_seen // 2 :])

    async def run(self, once: bool = False):
        self.is_running = True
        self.logger.info(
            "Heartbeat active. Interval=%ss Threshold=%.2f Limit=%d",
            self.interval,
            self.threshold,
            self.limit,
        )
        while self.is_running:
            try:
                await self.cycle()
            except Exception as exc:
                self.logger.error("Error in heartbeat cycle: %s", exc)
            if once:
                break
            await asyncio.sleep(self.interval)

    def stop(self):
        self.is_running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moltbook Heartbeat Daemon")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between scans")
    parser.add_argument("--threshold", type=float, default=30.0, help="Signal score threshold")
    parser.add_argument("--limit", type=int, default=20, help="Posts per scan")
    parser.add_argument("--max-seen", type=int, default=1000, help="Max dedup cache size")
    parser.add_argument("--log-file", type=str, default="nexus_heartbeat.log", help="Log file path")
    parser.add_argument("--no-memory", action="store_true", help="Disable MemoryForge writes")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow outbound network calls to Moltbook",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_network:
        print("ERROR network access requires --allow-network")
        return 2

    logger = build_logger(args.log_file)
    daemon = HeartbeatDaemon(
        interval=args.interval,
        threshold=args.threshold,
        limit=args.limit,
        max_seen=args.max_seen,
        use_memory=not args.no_memory,
        logger=logger,
    )

    try:
        asyncio.run(daemon.run(once=args.once))
    except KeyboardInterrupt:
        logger.info("Heartbeat stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
