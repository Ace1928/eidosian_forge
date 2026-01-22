import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_noCalls(self):
    """
        If there are no delayed calls, C{doIteration} is called with a
        timeout of L{None}.
        """
    reactor = TimeoutReportReactor()
    timeout = self._checkIterationTimeout(reactor)
    self.assertIsNone(timeout)