import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_onlyObject(self):
    """
        L{prefixedMethods} returns a list of the methods discovered on an
        object.
        """
    x = Base()
    output = prefixedMethods(x)
    self.assertEqual([x.method], output)