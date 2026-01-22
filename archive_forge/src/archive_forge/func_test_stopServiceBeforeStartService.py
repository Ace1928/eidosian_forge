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
def test_stopServiceBeforeStartService(self):
    """
        Calling L{ClientService.stopService} before
        L{ClientService.startService} returns a L{Deferred} that has
        already fired with L{None}.
        """
    clock = Clock()
    _, service = self.makeReconnector(fireImmediately=False, startService=False, clock=clock)
    d = service.stopService()
    self.assertIsNone(self.successResultOf(d))