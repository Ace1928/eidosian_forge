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
def test_unsetPid(self):
    """
        Test if pid is None/non-None before/after process termination.  This
        reuses process_echoer.py to get a process that blocks on stdin.
        """
    finished = defer.Deferred()
    p = TrivialProcessProtocol(finished)
    scriptPath = b'twisted.test.process_echoer'
    procTrans = reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv)
    self.assertTrue(procTrans.pid)

    def afterProcessEnd(ignored):
        self.assertIsNone(procTrans.pid)
    p.transport.closeStdin()
    return finished.addCallback(afterProcessEnd)