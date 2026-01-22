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
def test_killExitedButNotDetected(self):
    """
        L{process.Process.signalProcess} raises L{error.ProcessExitedAlready}
        if the process has exited but that twisted hasn't seen it (for example,
        if the process has been waited outside of twisted): C{os.kill} then
        raise C{OSError} with C{errno.ESRCH} as errno.
        """
    self.mockos.child = False
    self.mockos.waitChild = (0, 0)
    cmd = b'/mock/ouch'
    p = TrivialProcessProtocol(None)
    proc = reactor.spawnProcess(p, cmd, [b'ouch'], env=None, usePTY=False)
    self.mockos.raiseKill = OSError(errno.ESRCH, 'Not found')
    self.assertRaises(error.ProcessExitedAlready, proc.signalProcess, 'KILL')