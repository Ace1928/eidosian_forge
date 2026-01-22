import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_workingRepr(self):
    """
        L{reflect.safe_repr} produces the same output as C{repr} on a working
        object.
        """
    xs = ([1, 2, 3], b'a')
    self.assertEqual(list(map(reflect.safe_repr, xs)), list(map(repr, xs)))