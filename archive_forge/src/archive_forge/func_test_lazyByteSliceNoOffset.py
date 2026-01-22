import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_lazyByteSliceNoOffset(self):
    """
        L{lazyByteSlice} called with some bytes returns a semantically equal
        version of these bytes.
        """
    data = b'123XYZ'
    self.assertEqual(bytes(lazyByteSlice(data)), data)