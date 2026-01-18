"""Async publish/subscribe event bus for agent communication."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List


class EventBus:
    """Asynchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        """Initialize the bus with empty subscriber lists."""
        self._subscribers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = (
            defaultdict(list)
        )

    async def subscribe(
        self, event: str, handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """Register ``handler`` for ``event``."""
        self._subscribers[event].append(handler)

    async def unsubscribe(
        self, event: str, handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """Remove ``handler`` from ``event`` subscriptions."""
        if handler in self._subscribers.get(event, []):
            self._subscribers[event].remove(handler)

    async def publish(self, event: str, data: Any) -> None:
        """Publish ``data`` to all subscribers of ``event``."""
        await asyncio.gather(*(h(data) for h in list(self._subscribers.get(event, []))))
