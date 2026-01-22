from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings
def test_native_coroutine_timeout(self):

    @gen_test(timeout=0.1)
    async def test(self):
        await gen.sleep(1)
    try:
        test(self)
        self.fail('did not get expected exception')
    except ioloop.TimeoutError:
        self.finished = True