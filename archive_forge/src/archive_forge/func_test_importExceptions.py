import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_importExceptions(self):
    """
        Exceptions raised by modules which L{namedAny} causes to be imported
        should pass through L{namedAny} to the caller.
        """
    self.assertRaises(ZeroDivisionError, reflect.namedAny, 'twisted.test.reflect_helper_ZDE')
    self.assertRaises(ZeroDivisionError, reflect.namedAny, 'twisted.test.reflect_helper_ZDE')
    self.assertRaises(ValueError, reflect.namedAny, 'twisted.test.reflect_helper_VE')
    self.assertRaises(ImportError, reflect.namedAny, 'twisted.test.reflect_helper_IE')