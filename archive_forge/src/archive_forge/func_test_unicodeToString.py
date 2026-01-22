import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_unicodeToString(self):
    """
        C{nativeString} converts unicode to the native string format, assuming
        an ASCII encoding if applicable.
        """
    self.assertNativeString('Good day', 'Good day')