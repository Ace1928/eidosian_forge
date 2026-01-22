from __future__ import annotations
from collections.abc import Callable
from contextlib import contextmanager
from typing import ClassVar
@contextmanager
def local_callbacks(callbacks=None):
    """Allows callbacks to work with nested schedulers.

    Callbacks will only be used by the first started scheduler they encounter.
    This means that only the outermost scheduler will use global callbacks."""
    global_callbacks = callbacks is None
    if global_callbacks:
        callbacks, Callback.active = (Callback.active, set())
    try:
        yield (callbacks or ())
    finally:
        if global_callbacks:
            Callback.active = callbacks