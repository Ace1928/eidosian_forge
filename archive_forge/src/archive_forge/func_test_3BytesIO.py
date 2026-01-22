import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_3BytesIO(self):
    """
        An L{io.BytesIO} accepts and returns bytes.
        """
    self.assertEqual(ioType(io.BytesIO()), bytes)