import asyncio
import threading
from typing import List, Union, Any, TypeVar, Generic, Optional, Callable, Awaitable
from unittest.mock import AsyncMock
def wire_queues(mock: AsyncMock) -> QueuePair:
    queues = QueuePair()
    mock.side_effect = make_queue_waiter(queues.called, queues.results)
    return queues