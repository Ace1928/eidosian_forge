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
def test_killErrorInKill(self):
    """
        L{process.Process.signalProcess} doesn't mask C{OSError} exceptions if
        the errno is different from C{errno.ESRCH}.
        """
    self.mockos.child = False
    self.mockos.waitChild = (0, 0)
    cmd = b'/mock/ouch'
    p = TrivialProcessProtocol(None)
    proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
    self.mockos.raiseKill = OSError(errno.EINVAL, 'Invalid signal')
    err = self.assertRaises(OSError, proc.signalProcess, 'KILL')
    self.assertEqual(err.errno, errno.EINVAL)