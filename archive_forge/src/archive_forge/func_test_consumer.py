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
def test_consumer(self):
    """
        Verify that the transport of a protocol connected to L{StandardIO}
        is a working L{IConsumer} provider.
        """
    p = StandardIOTestProcessProtocol()
    d = p.onCompletion
    junkPath = self._junkPath()
    self._spawnProcess(p, b'stdio_test_consumer', junkPath)

    def processEnded(reason):
        with open(junkPath, 'rb') as f:
            self.assertEqual(p.data[1], f.read())
        reason.trap(error.ProcessDone)
    return self._requireFailure(d, processEnded)