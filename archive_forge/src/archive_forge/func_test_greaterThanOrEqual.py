import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_greaterThanOrEqual(self):
    """
        Instances of a class that is decorated by C{comparable} support
        greater-than-or-equal comparisons.
        """
    self.assertTrue(Comparable(1) >= Comparable(1))
    self.assertTrue(Comparable(2) >= Comparable(1))
    self.assertFalse(Comparable(0) >= Comparable(3))