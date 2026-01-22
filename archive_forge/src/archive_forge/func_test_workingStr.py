import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_workingStr(self):
    x = [1, 2, 3]
    self.assertEqual(reflect.safe_str(x), str(x))