from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
import attrs
from .. import _core
from .._deprecate import deprecated
from .._util import final
def get_batch_nowait(self) -> list[T]:
    """Attempt to get the next batch from the queue, without blocking.

        Returns:
          list: A list of dequeued items, in order. On a successful call this
              list is always non-empty; if it would be empty we raise
              :exc:`~trio.WouldBlock` instead.

        Raises:
          ~trio.WouldBlock: if the queue is empty.

        """
    if not self._can_get:
        raise _core.WouldBlock
    return self._get_batch_protected()