from __future__ import annotations
import signal
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING
import trio
from ._util import ConflictDetector, is_main_thread, signal_raise
def get_pending_signal_count(rec: AsyncIterator[int]) -> int:
    """Helper for tests, not public or otherwise used."""
    assert isinstance(rec, SignalReceiver)
    return len(rec._pending)