import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_lazyByteSliceOffsetAndLength(self):
    """
        L{lazyByteSlice} called with some bytes, an offset and a length returns
        a semantically equal version of these bytes starting at the given
        offset, up to the given length.
        """
    data = b'123XYZ'
    self.assertEqual(bytes(lazyByteSlice(data, 2, 3)), data[2:5])