import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def test_mockForkErrorClosePTY(self):
    """
        When C{os.fork} raises an exception, the file descriptors created by
        C{pty.openpty} are closed and don't leak, when C{usePTY} is set to
        C{True}.
        """
    self.mockos.raiseFork = OSError(errno.EAGAIN, None)
    protocol = TrivialProcessProtocol(None)
    self.assertRaises(OSError, reactor.spawnProcess, protocol, None, usePTY=True)
    self.assertEqual(self.mockos.actions, [('fork', False)])
    self.assertEqual(set(self.mockos.closed), {-12, -13})