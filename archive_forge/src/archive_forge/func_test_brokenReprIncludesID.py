import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_brokenReprIncludesID(self):
    """
        C{id} is used to print the ID of the object in case of an error.

        L{safe_repr} includes a traceback after a newline, so we only check
        against the first line of the repr.
        """

    class X(BTBase):
        breakRepr = True
    xRepr = reflect.safe_repr(X)
    xReprExpected = f'<BrokenType instance at 0x{id(X):x} with repr error:'
    self.assertEqual(xReprExpected, xRepr.split('\n')[0])