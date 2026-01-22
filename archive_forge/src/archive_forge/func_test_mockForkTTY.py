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
def test_mockForkTTY(self):
    """
        Test a TTY spawnProcess: check the path of the client code:
        fork, exec, exit.
        """
    cmd = b'/mock/ouch'
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    self.assertRaises(SystemError, reactor.spawnProcess, p, cmd, [b'ouch'], env=None, usePTY=True)
    self.assertTrue(self.mockos.exited)
    self.assertEqual(self.mockos.actions, [('fork', False), 'setsid', 'exec', ('exit', 1)])