from __future__ import annotations
import errno
import select
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, Literal
import attrs
import outcome
from .. import _core
from ._run import _public
from ._wakeup_socketpair import WakeupSocketpair
@contextmanager
@_public
def monitor_kevent(self, ident: int, filter: int) -> Iterator[_core.UnboundedQueue[select.kevent]]:
    """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__.
        """
    key = (ident, filter)
    if key in self._registered:
        raise _core.BusyResourceError('attempt to register multiple listeners for same ident/filter pair')
    q = _core.UnboundedQueue[select.kevent]()
    self._registered[key] = q
    try:
        yield q
    finally:
        del self._registered[key]