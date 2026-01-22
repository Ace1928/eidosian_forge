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
def test_already_done(self):
    f1 = Future()
    f2 = Future()
    f3 = Future()
    f1.set_result(24)
    f2.set_result(42)
    f3.set_result(84)
    g = gen.WaitIterator(f1, f2, f3)
    i = 0
    while not g.done():
        r = (yield g.next())
        if i == 0:
            self.assertEqual(g.current_index, 0)
            self.assertIs(g.current_future, f1)
            self.assertEqual(r, 24)
        elif i == 1:
            self.assertEqual(g.current_index, 1)
            self.assertIs(g.current_future, f2)
            self.assertEqual(r, 42)
        elif i == 2:
            self.assertEqual(g.current_index, 2)
            self.assertIs(g.current_future, f3)
            self.assertEqual(r, 84)
        i += 1
    self.assertEqual(g.current_index, None, 'bad nil current index')
    self.assertEqual(g.current_future, None, 'bad nil current future')
    dg = gen.WaitIterator(f1=f1, f2=f2)
    while not dg.done():
        dr = (yield dg.next())
        if dg.current_index == 'f1':
            self.assertTrue(dg.current_future == f1 and dr == 24, 'WaitIterator dict status incorrect')
        elif dg.current_index == 'f2':
            self.assertTrue(dg.current_future == f2 and dr == 42, 'WaitIterator dict status incorrect')
        else:
            self.fail('got bad WaitIterator index {}'.format(dg.current_index))
        i += 1
    self.assertEqual(dg.current_index, None, 'bad nil current index')
    self.assertEqual(dg.current_future, None, 'bad nil current future')