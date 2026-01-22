import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_3openBinaryMode(self):
    """
        A file opened via 'io.open' in binary mode accepts and returns bytes.
        """
    with open(self.mktemp(), 'wb') as f:
        self.assertEqual(ioType(f), bytes)