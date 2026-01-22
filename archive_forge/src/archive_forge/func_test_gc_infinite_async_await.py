import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
def test_gc_infinite_async_await(self):
    import asyncio

    async def infinite_coro(result):
        try:
            while True:
                await gen.sleep(0.001)
                result.append(True)
        finally:
            result.append(None)
    loop = self.get_new_ioloop()
    result = []
    wfut = []

    @gen.coroutine
    def do_something():
        fut = asyncio.get_event_loop().create_task(infinite_coro(result))
        fut._refcycle = fut
        wfut.append(weakref.ref(fut))
        yield gen.sleep(0.2)
    loop.run_sync(do_something)
    with ExpectLog('asyncio', 'Task was destroyed but it is pending'):
        loop.close()
        gc.collect()
    self.assertIs(wfut[0](), None)
    self.assertGreaterEqual(len(result), 2)
    if not self.is_pypy3():
        self.assertIs(result[-1], None)