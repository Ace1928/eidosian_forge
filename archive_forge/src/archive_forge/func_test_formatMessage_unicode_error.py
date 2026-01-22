import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def test_formatMessage_unicode_error(self):
    one = ''.join((chr(i) for i in range(255)))
    self.testableTrue._formatMessage(one, '�')