from __future__ import annotations
import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any, ClassVar
from weakref import WeakValueDictionary
def combine_locks(locks):
    """Combine a sequence of locks into a single lock."""
    all_locks = []
    for lock in locks:
        if isinstance(lock, CombinedLock):
            all_locks.extend(lock.locks)
        elif lock is not None:
            all_locks.append(lock)
    num_locks = len(all_locks)
    if num_locks > 1:
        return CombinedLock(all_locks)
    elif num_locks == 1:
        return all_locks[0]
    else:
        return DummyLock()