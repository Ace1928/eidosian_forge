import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_delayDelayedCall(self):
    """
        If a delayed call is re-delayed, the timeout passed to
        C{doIteration} is based on the remaining time before the call would
        have been made and the additional amount of time passed to the delay
        method.
        """
    reactor = TimeoutReportReactor()
    call = reactor.callLater(50, lambda: None)
    reactor.now += 10
    call.delay(20)
    timeout = self._checkIterationTimeout(reactor)
    self.assertEqual(timeout, 60)