from . import events
from . import futures
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import typing
def uncancel(self) -> None:
    raise NotImplementedError('QtTask.uncancel is not implemented')