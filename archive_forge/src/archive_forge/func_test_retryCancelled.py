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
def test_retryCancelled(self):
    """
        When L{ClientService.stopService} is called while waiting between
        connection attempts, the pending reconnection attempt is cancelled and
        the service is stopped immediately.
        """
    clock = Clock()
    cq, service = self.makeReconnector(fireImmediately=False, clock=clock)
    cq.connectQueue[0].errback(Exception('no connection'))
    d = service.stopService()
    self.assertEqual(clock.getDelayedCalls(), [])
    self.successResultOf(d)