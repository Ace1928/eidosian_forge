import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
def test_processExited(self):
    """
        L{IProcessProtocol.processExited} is called when the child process
        exits, even if file descriptors associated with the child are still
        open.
        """
    exited = Deferred()
    allLost = Deferred()
    lost = []

    class Waiter(ProcessProtocol):

        def childDataReceived(self, fd, data):
            msg('childDataReceived(%d, %r)' % (fd, data))

        def childConnectionLost(self, childFD):
            msg('childConnectionLost(%d)' % (childFD,))
            lost.append(childFD)
            if len(lost) == 3:
                allLost.callback(None)

        def processExited(self, reason):
            msg(f'processExited({reason!r})')
            exited.callback([reason])
            self.transport.loseConnection()
    reactor = self.buildReactor()
    reactor.callWhenRunning(reactor.spawnProcess, Waiter(), pyExe, [pyExe, b'-u', b'-m', self.keepStdioOpenProgram, b'child', self.keepStdioOpenArg], env=properEnv, usePTY=self.usePTY)

    def cbExited(args):
        failure, = args
        failure.trap(ProcessDone)
        msg(f'cbExited; lost = {lost}')
        self.assertEqual(lost, [])
        return allLost
    exited.addCallback(cbExited)

    def cbAllLost(ignored):
        self.assertEqual(set(lost), {0, 1, 2})
    exited.addCallback(cbAllLost)
    exited.addErrback(err)
    exited.addCallback(lambda ign: reactor.stop())
    self.runReactor(reactor)