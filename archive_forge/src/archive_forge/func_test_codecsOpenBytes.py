import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_codecsOpenBytes(self):
    """
        The L{codecs} module, oddly, returns a file-like object which returns
        bytes when not passed an 'encoding' argument.
        """
    with codecs.open(self.mktemp(), 'wb') as f:
        self.assertEqual(ioType(f), bytes)