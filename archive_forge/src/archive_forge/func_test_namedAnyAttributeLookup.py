import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_namedAnyAttributeLookup(self):
    """
        L{namedAny} should return the object an attribute of a non-module,
        non-package object is bound to for the name it is passed.
        """
    self.assertEqual(reflect.namedAny('twisted.test.test_reflect.Summer.reallySet'), Summer.reallySet)