from asyncio import Future, Queue, ensure_future, sleep
from inspect import isawaitable
from typing import Any, AsyncIterator, Callable, Optional, Set
def get_subscriber(self, transform: Optional[Callable]=None) -> 'SimplePubSubIterator':
    return SimplePubSubIterator(self, transform)