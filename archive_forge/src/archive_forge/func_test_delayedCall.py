import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_delayedCall(self):
    """
        If there is a delayed call, C{doIteration} is called with a timeout
        which is the difference between the current time and the time at
        which that call is to run.
        """
    reactor = TimeoutReportReactor()
    reactor.callLater(100, lambda: None)
    timeout = self._checkIterationTimeout(reactor)
    self.assertEqual(timeout, 100)