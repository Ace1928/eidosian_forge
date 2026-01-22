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
def test_errorInProcessEnded(self):
    """
        The handler which reaps a process is removed when the process is
        reaped, even if the protocol's C{processEnded} method raises an
        exception.
        """
    connected = defer.Deferred()
    ended = defer.Deferred()
    scriptPath = b'twisted.test.process_echoer'

    class ErrorInProcessEnded(protocol.ProcessProtocol):
        """
            A protocol that raises an error in C{processEnded}.
            """

        def makeConnection(self, transport):
            connected.callback(transport)

        def processEnded(self, reason):
            reactor.callLater(0, ended.callback, None)
            raise RuntimeError('Deliberate error')
    reactor.spawnProcess(ErrorInProcessEnded(), pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv, path=None)
    pid = []

    def cbConnected(transport):
        pid.append(transport.pid)
        self.assertIn(transport.pid, process.reapProcessHandlers)
        transport.loseConnection()
    connected.addCallback(cbConnected)

    def checkTerminated(ignored):
        excs = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(excs), 1)
        self.assertNotIn(pid[0], process.reapProcessHandlers)
    ended.addCallback(checkTerminated)
    return ended