import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_startStopFactory(self):
    """
        Although somewhat obscure, L{IProtocolFactory} includes both C{doStart}
        and C{doStop} methods; ensure that when these methods are called on the
        factory that was passed to the reactor, the factory that was passed
        from the application receives them.
        """
    cq, service = self.makeReconnector()
    firstAppFactory = cq.applicationFactory
    self.assertEqual(firstAppFactory.numPorts, 0)
    firstPassedFactory = cq.passedFactories[0]
    firstPassedFactory.doStart()
    self.assertEqual(firstAppFactory.numPorts, 1)