from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
@contextmanager
@_public
def monitor_completion_key(self) -> Iterator[tuple[int, UnboundedQueue[object]]]:
    """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__ and `#52
        <https://github.com/python-trio/trio/issues/52>`__.
        """
    key = next(self._completion_key_counter)
    queue = _core.UnboundedQueue[object]()
    self._completion_key_queues[key] = queue
    try:
        yield (key, queue)
    finally:
        del self._completion_key_queues[key]