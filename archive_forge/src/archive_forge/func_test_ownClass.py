import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_ownClass(self):
    """
        If x is and instance of Base and Base defines a method named method,
        L{accumulateMethods} adds an item to the given dictionary with
        C{"method"} as the key and a bound method object for Base.method value.
        """
    x = Base()
    output = {}
    accumulateMethods(x, output)
    self.assertEqual({'method': x.method}, output)