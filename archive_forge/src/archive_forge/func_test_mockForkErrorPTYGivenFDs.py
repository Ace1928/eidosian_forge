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
def test_mockForkErrorPTYGivenFDs(self):
    """
        If a tuple is passed to C{usePTY} to specify slave and master file
        descriptors and that C{os.fork} raises an exception, these file
        descriptors aren't closed.
        """
    self.mockos.raiseFork = OSError(errno.EAGAIN, None)
    protocol = TrivialProcessProtocol(None)
    self.assertRaises(OSError, reactor.spawnProcess, protocol, None, usePTY=(-20, -21, 'foo'))
    self.assertEqual(self.mockos.actions, [('fork', False)])
    self.assertEqual(self.mockos.closed, [])