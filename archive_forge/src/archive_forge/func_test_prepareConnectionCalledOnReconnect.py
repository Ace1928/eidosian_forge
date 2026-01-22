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
def test_prepareConnectionCalledOnReconnect(self):
    """
        The C{prepareConnection} callback is invoked each time a connection is
        made, including on reconnection.
        """
    prepares = [0]

    def prepareConnection(_proto):
        prepares[0] += 1
    clock = Clock()
    cq, service = self.makeReconnector(prepareConnection=prepareConnection, clock=clock)
    self.assertEqual(1, prepares[0])
    cq.constructedProtocols[0].connectionLost(Failure(IndentationError()))
    clock.advance(AT_LEAST_ONE_ATTEMPT)
    self.assertEqual(2, prepares[0])