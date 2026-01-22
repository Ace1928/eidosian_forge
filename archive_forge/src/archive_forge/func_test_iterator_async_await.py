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
@gen_test
def test_iterator_async_await(self):
    futures = [Future(), Future(), Future(), Future()]
    self.finish_coroutines(0, futures)
    self.finished = False

    async def f():
        i = 0
        g = gen.WaitIterator(*futures)
        try:
            async for r in g:
                if i == 0:
                    self.assertEqual(r, 24, 'iterator value incorrect')
                    self.assertEqual(g.current_index, 2, 'wrong index')
                else:
                    raise Exception('expected exception on iteration 1')
                i += 1
        except ZeroDivisionError:
            i += 1
        async for r in g:
            if i == 2:
                self.assertEqual(r, 42, 'iterator value incorrect')
                self.assertEqual(g.current_index, 1, 'wrong index')
            elif i == 3:
                self.assertEqual(r, 84, 'iterator value incorrect')
                self.assertEqual(g.current_index, 3, 'wrong index')
            else:
                raise Exception("didn't expect iteration %d" % i)
            i += 1
        self.finished = True
    yield f()
    self.assertTrue(self.finished)