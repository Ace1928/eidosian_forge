import asyncio
import inspect
from asyncio import Future
from functools import wraps
from types import CoroutineType
from typing import (
from twisted.internet import defer
from twisted.internet.defer import Deferred, DeferredList, ensureDeferred
from twisted.internet.task import Cooperator
from twisted.python import failure
from twisted.python.failure import Failure
from scrapy.exceptions import IgnoreRequest
from scrapy.utils.reactor import _get_asyncio_event_loop, is_asyncio_reactor_installed
def maybeDeferred_coro(f: Callable, *args: Any, **kw: Any) -> Deferred:
    """Copy of defer.maybeDeferred that also converts coroutines to Deferreds."""
    try:
        result = f(*args, **kw)
    except:
        return defer.fail(failure.Failure(captureVars=Deferred.debug))
    if isinstance(result, Deferred):
        return result
    if asyncio.isfuture(result) or inspect.isawaitable(result):
        return deferred_from_coro(result)
    if isinstance(result, failure.Failure):
        return defer.fail(result)
    return defer.succeed(result)