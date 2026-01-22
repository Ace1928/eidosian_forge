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
def test_whenConnectedFails(self):
    """
        L{ClientService.whenConnected} returns a L{Deferred} that fails, if
        asked, when some number of connections have failed.
        """
    clock = Clock()
    cq, service = self.makeReconnector(fireImmediately=False, clock=clock)
    a0 = service.whenConnected()
    a1 = service.whenConnected(failAfterFailures=1)
    a2 = service.whenConnected(failAfterFailures=2)
    a3 = service.whenConnected(failAfterFailures=3)
    self.assertNoResult(a0)
    self.assertNoResult(a1)
    self.assertNoResult(a2)
    self.assertNoResult(a3)
    f1 = Failure(Exception())
    cq.connectQueue[0].errback(f1)
    self.assertNoResult(a0)
    self.assertIdentical(self.failureResultOf(a1, Exception), f1)
    self.assertNoResult(a2)
    self.assertNoResult(a3)
    clock.advance(AT_LEAST_ONE_ATTEMPT)
    self.assertEqual(len(cq.connectQueue), 2)
    self.assertNoResult(a0)
    self.assertNoResult(a2)
    self.assertNoResult(a3)
    f2 = Failure(Exception())
    cq.connectQueue[1].errback(f2)
    self.assertNoResult(a0)
    self.assertIdentical(self.failureResultOf(a2, Exception), f2)
    self.assertNoResult(a3)
    AT_LEAST_TWO_ATTEMPTS = AT_LEAST_ONE_ATTEMPT
    clock.advance(AT_LEAST_TWO_ATTEMPTS)
    self.assertEqual(len(cq.connectQueue), 3)
    self.assertNoResult(a0)
    self.assertNoResult(a3)
    cq.connectQueue[2].callback(None)
    resultA0 = self.successResultOf(a0)
    resultA3 = self.successResultOf(a3)
    self.assertIdentical(resultA0, resultA3)
    self.assertIdentical(resultA0, cq.applicationProtocols[0])
    a4 = service.whenConnected(failAfterFailures=1)
    resultA4 = self.successResultOf(a4)
    self.assertIdentical(resultA0, resultA4)