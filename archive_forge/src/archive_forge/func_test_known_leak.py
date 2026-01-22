import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython
def test_known_leak(self):

    class C(object):

        def __init__(self, name):
            self.name = name
            self.a: typing.Optional[C] = None
            self.b: typing.Optional[C] = None
            self.c: typing.Optional[C] = None

        def __repr__(self):
            return f'name={self.name}'
    with self.assertRaises(AssertionError) as cm:
        with assert_no_cycle_garbage():
            a = C('a')
            b = C('b')
            c = C('c')
            a.b = b
            a.c = c
            b.a = a
            b.c = c
            del a, b
    self.assertIn('Circular', str(cm.exception))
    self.assertIn('    name=a', str(cm.exception))
    self.assertIn('    name=b', str(cm.exception))
    self.assertNotIn('    name=c', str(cm.exception))