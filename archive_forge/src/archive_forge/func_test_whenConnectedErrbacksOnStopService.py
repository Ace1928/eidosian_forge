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
def test_whenConnectedErrbacksOnStopService(self):
    """
        L{ClientService.whenConnected} returns a L{Deferred} that
        errbacks with L{CancelledError} if
        L{ClientService.stopService} is called between connection
        attempts.
        """
    clock = Clock()
    cq, service = self.makeReconnector(fireImmediately=False, clock=clock)
    beforeErrbackAndStop = service.whenConnected()
    cq.connectQueue[0].errback(Exception('no connection'))
    service.stopService()
    afterErrbackAndStop = service.whenConnected()
    self.assertIsInstance(self.failureResultOf(beforeErrbackAndStop).value, CancelledError)
    self.assertIsInstance(self.failureResultOf(afterErrbackAndStop).value, CancelledError)