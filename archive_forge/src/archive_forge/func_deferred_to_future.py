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
def deferred_to_future(d: Deferred) -> Future:
    """
    .. versionadded:: 2.6.0

    Return an :class:`asyncio.Future` object that wraps *d*.

    When :ref:`using the asyncio reactor <install-asyncio>`, you cannot await
    on :class:`~twisted.internet.defer.Deferred` objects from :ref:`Scrapy
    callables defined as coroutines <coroutine-support>`, you can only await on
    ``Future`` objects. Wrapping ``Deferred`` objects into ``Future`` objects
    allows you to wait on them::

        class MySpider(Spider):
            ...
            async def parse(self, response):
                additional_request = scrapy.Request('https://example.org/price')
                deferred = self.crawler.engine.download(additional_request)
                additional_response = await deferred_to_future(deferred)
    """
    return d.asFuture(_get_asyncio_event_loop())