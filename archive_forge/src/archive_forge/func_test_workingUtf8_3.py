import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_workingUtf8_3(self):
    """
        L{safe_str} for C{bytes} with utf-8 encoded data should return
        the value decoded into C{str}.
        """
    x = b't\xc3\xbcst'
    self.assertEqual(reflect.safe_str(x), x.decode('utf-8'))