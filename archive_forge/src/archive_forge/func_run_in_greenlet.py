from __future__ import annotations
import asyncio  # noqa
import typing
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import TypeVar
def run_in_greenlet(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run sync function in greenlet. Support nested calls"""
    if have_greenlet:
        if self.runner.get_loop().is_running():
            return fn(*args, **kwargs)
        else:
            return self.runner.run(greenlet_spawn(fn, *args, **kwargs))
    else:
        return fn(*args, **kwargs)