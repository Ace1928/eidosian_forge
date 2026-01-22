from __future__ import annotations
import codecs
import queue
import threading
from typing import Iterator, List, Optional, cast
from ..frames import Frame, Opcode
from ..typing import Data
def get_iter(self) -> Iterator[Data]:
    """
        Stream the next message.

        Iterating the return value of :meth:`get_iter` yields a :class:`str` or
        :class:`bytes` for each frame in the message.

        The iterator must be fully consumed before calling :meth:`get_iter` or
        :meth:`get` again. Else, :exc:`RuntimeError` is raised.

        This method only makes sense for fragmented messages. If messages aren't
        fragmented, use :meth:`get` instead.

        Raises:
            EOFError: If the stream of frames has ended.
            RuntimeError: If two threads run :meth:`get` or :meth:``get_iter`
                concurrently.

        """
    with self.mutex:
        if self.closed:
            raise EOFError('stream of frames ended')
        if self.get_in_progress:
            raise RuntimeError('get or get_iter is already running')
        chunks = self.chunks
        self.chunks = []
        self.chunks_queue = cast('queue.SimpleQueue[Optional[Data]]', queue.SimpleQueue())
        if self.message_complete.is_set():
            self.chunks_queue.put(None)
        self.get_in_progress = True
    yield from chunks
    while True:
        chunk = self.chunks_queue.get()
        if chunk is None:
            break
        yield chunk
    with self.mutex:
        self.get_in_progress = False
        assert self.message_complete.is_set()
        self.message_complete.clear()
        if self.closed:
            raise EOFError('stream of frames ended')
        assert not self.message_fetched.is_set()
        self.message_fetched.set()
        assert self.chunks == []
        self.chunks_queue = None