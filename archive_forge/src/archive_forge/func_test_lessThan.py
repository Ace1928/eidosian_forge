import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_lessThan(self):
    """
        L{cmp} returns -1 if its first argument is smaller than its second.
        """
    self.assertEqual(cmp(0.1, 2.3), -1)
    self.assertEqual(cmp(b'a', b'd'), -1)