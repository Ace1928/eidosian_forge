import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_inheritedMethod(self):
    """
        L{prefixedMethodNames} returns a list included methods with the given
        prefix defined on base classes of the class passed to it.
        """

    class Child(Separate):
        pass
    self.assertEqual(['method'], prefixedMethodNames(Child, 'good_'))