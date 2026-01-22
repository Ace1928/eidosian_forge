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
def test_clientConnectionLost(self):
    """
        When a client connection is lost, the service removes its reference
        to the protocol and calls retry.
        """
    clock = Clock()
    cq, service = self.makeReconnector(clock=clock, fireImmediately=False)
    self.assertEqual(len(cq.connectQueue), 1)
    cq.connectQueue[0].callback(None)
    self.assertEqual(len(cq.connectQueue), 1)
    self.assertIdentical(self.successResultOf(service.whenConnected()), cq.applicationProtocols[0])
    cq.constructedProtocols[0].connectionLost(Failure(Exception()))
    clock.advance(AT_LEAST_ONE_ATTEMPT)
    self.assertEqual(len(cq.connectQueue), 2)
    cq.connectQueue[1].callback(None)
    self.assertIdentical(self.successResultOf(service.whenConnected()), cq.applicationProtocols[1])