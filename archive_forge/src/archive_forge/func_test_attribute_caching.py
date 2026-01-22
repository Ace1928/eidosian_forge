import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_attribute_caching(self):

    class A(object):

        def __init__(self):
            self.call_counter = 0

        @misc.cachedproperty
        def b(self):
            self.call_counter += 1
            return 'b'
    a = A()
    self.assertEqual('b', a.b)
    self.assertEqual('b', a.b)
    self.assertEqual(1, a.call_counter)