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
def test_abnormalTermination(self):
    """
        When a process terminates with a system exit code set to 1,
        C{processEnded} is called with a L{error.ProcessTerminated} error,
        the C{exitCode} attribute reflecting the system exit code.
        """
    d = defer.Deferred()
    p = TrivialProcessProtocol(d)
    reactor.spawnProcess(p, pyExe, [pyExe, b'-c', b'import sys; sys.exit(1)'], env=None, usePTY=self.usePTY)

    def check(ignored):
        p.reason.trap(error.ProcessTerminated)
        self.assertEqual(p.reason.value.exitCode, 1)
        self.assertIsNone(p.reason.value.signal)
    d.addCallback(check)
    return d