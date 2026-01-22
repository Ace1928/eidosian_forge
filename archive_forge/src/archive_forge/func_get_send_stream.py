import asyncio
from asyncio import AbstractEventLoop, Queue
from typing import AsyncIterator, Generic, TypeVar
def get_send_stream(self) -> _SendStream[T]:
    """Get a writer for the channel."""
    return _SendStream[T](reader_loop=self._loop, queue=self._queue, done=self._done)