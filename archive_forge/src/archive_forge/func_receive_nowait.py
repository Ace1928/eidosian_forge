from __future__ import annotations
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar
from .. import (
from ..abc import Event, ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint
def receive_nowait(self) -> T_co:
    """
        Receive the next item if it can be done without waiting.

        :return: the received item
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.EndOfStream: if the buffer is empty and this stream has been
            closed from the sending end
        :raises ~anyio.WouldBlock: if there are no items in the buffer and no tasks
            waiting to send

        """
    if self._closed:
        raise ClosedResourceError
    if self._state.waiting_senders:
        send_event, item = self._state.waiting_senders.popitem(last=False)
        self._state.buffer.append(item)
        send_event.set()
    if self._state.buffer:
        return self._state.buffer.popleft()
    elif not self._state.open_send_channels:
        raise EndOfStream
    raise WouldBlock