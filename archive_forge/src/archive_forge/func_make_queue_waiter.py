import asyncio
import threading
from typing import List, Union, Any, TypeVar, Generic, Optional, Callable, Awaitable
from unittest.mock import AsyncMock
def make_queue_waiter(started_q: 'asyncio.Queue[None]', result_q: 'asyncio.Queue[Union[T, Exception]]') -> Callable[[], Awaitable[T]]:
    """
    Given a queue to notify when started and a queue to get results from, return a waiter which
    notifies started_q when started and returns from result_q when done.
    """

    async def waiter(*args, **kwargs):
        await started_q.put(None)
        result = await result_q.get()
        if isinstance(result, Exception):
            raise result
        return result
    return waiter