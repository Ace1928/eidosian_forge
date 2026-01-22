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
def test_killExited(self):
    """
        L{process.Process.signalProcess} raises L{error.ProcessExitedAlready}
        if the process has exited.
        """
    self.mockos.child = False
    cmd = b'/mock/ouch'
    p = TrivialProcessProtocol(None)
    proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
    self.assertRaises(error.ProcessExitedAlready, proc.signalProcess, 'KILL')