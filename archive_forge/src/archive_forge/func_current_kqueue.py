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
@_public
def current_kqueue(self) -> select.kqueue:
    """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__.
        """
    return self._kqueue