import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def test_formatMsg(self):
    self.assertEqual(self.testableFalse._formatMessage(None, 'foo'), 'foo')
    self.assertEqual(self.testableFalse._formatMessage('foo', 'bar'), 'foo')
    self.assertEqual(self.testableTrue._formatMessage(None, 'foo'), 'foo')
    self.assertEqual(self.testableTrue._formatMessage('foo', 'bar'), 'bar : foo')
    self.testableTrue._formatMessage(object(), 'foo')