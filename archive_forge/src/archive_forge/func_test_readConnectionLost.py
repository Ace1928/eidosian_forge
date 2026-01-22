import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
def test_readConnectionLost(self):
    """
        When stdin is closed and the protocol connected to it implements
        L{IHalfCloseableProtocol}, the protocol's C{readConnectionLost} method
        is called.
        """
    errorLogFile = self.mktemp()
    log.msg('Child process logging to ' + errorLogFile)
    p = StandardIOTestProcessProtocol()
    p.onDataReceived = defer.Deferred()

    def cbBytes(ignored):
        d = p.onCompletion
        p.transport.closeStdin()
        return d
    p.onDataReceived.addCallback(cbBytes)

    def processEnded(reason):
        reason.trap(error.ProcessDone)
    d = self._requireFailure(p.onDataReceived, processEnded)
    self._spawnProcess(p, b'stdio_test_halfclose', errorLogFile)
    return d