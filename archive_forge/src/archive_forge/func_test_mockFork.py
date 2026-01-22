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
def test_mockFork(self):
    """
        Test a classic spawnProcess. Check the path of the client code:
        fork, exec, exit.
        """
    gc.enable()
    cmd = b'/mock/ouch'
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    try:
        reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
    except SystemError:
        self.assertTrue(self.mockos.exited)
        self.assertEqual(self.mockos.actions, [('fork', False), 'exec', ('exit', 1)])
    else:
        self.fail('Should not be here')
    self.assertFalse(gc.isenabled())