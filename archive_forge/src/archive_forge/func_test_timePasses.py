import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_timePasses(self):
    """
        If a delayed call is scheduled and then some time passes, the
        timeout passed to C{doIteration} is reduced by the amount of time
        which passed.
        """
    reactor = TimeoutReportReactor()
    reactor.callLater(100, lambda: None)
    reactor.now += 25
    timeout = self._checkIterationTimeout(reactor)
    self.assertEqual(timeout, 75)