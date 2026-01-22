from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Callable
def success_handler(_: Any) -> None:
    loop.call_soon_threadsafe(asyncio_future.set_result, response_future.result())