from __future__ import annotations
import asyncio
import atexit
import errno
import inspect
import sys
import threading
import warnings
from contextvars import ContextVar
from pathlib import Path
from types import FrameType
from typing import Any, Awaitable, Callable, TypeVar, cast
def ensure_event_loop(prefer_selector_loop: bool=False) -> asyncio.AbstractEventLoop:
    loop = _loop.get()
    if loop is not None and (not loop.is_closed()):
        return loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if sys.platform == 'win32' and prefer_selector_loop:
            loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    _loop.set(loop)
    return loop