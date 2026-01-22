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
def process_chain_both(callbacks: Iterable[Callable], errbacks: Iterable[Callable], input: Any, *a: Any, **kw: Any) -> Deferred:
    """Return a Deferred built by chaining the given callbacks and errbacks"""
    d: Deferred = Deferred()
    for cb, eb in zip(callbacks, errbacks):
        d.addCallbacks(callback=cb, errback=eb, callbackArgs=a, callbackKeywords=kw, errbackArgs=a, errbackKeywords=kw)
    if isinstance(input, failure.Failure):
        d.errback(input)
    else:
        d.callback(input)
    return d