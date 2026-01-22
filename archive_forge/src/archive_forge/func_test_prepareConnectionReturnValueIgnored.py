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
def test_prepareConnectionReturnValueIgnored(self):
    """
        The C{prepareConnection} return value is ignored when it does not
        indicate a failure. Even though the callback participates in the
        internal new-connection L{Deferred} chain for error propagation
        purposes, any successful result does not affect the ultimate return
        value.
        """
    sentinel = object()

    def prepareConnection(proto):
        return sentinel
    cq, service = self.makeReconnector(prepareConnection=prepareConnection)
    result = self.successResultOf(service.whenConnected())
    self.assertNotIdentical(sentinel, result)