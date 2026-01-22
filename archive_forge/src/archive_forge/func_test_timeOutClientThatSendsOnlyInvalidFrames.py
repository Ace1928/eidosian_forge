import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def test_timeOutClientThatSendsOnlyInvalidFrames(self):
    """
        A client that sends only invalid frames is eventually timed out.
        """
    memoryReactor = MemoryReactorClock()
    connection = H2Connection(memoryReactor)
    connection.callLater = memoryReactor.callLater
    connection.timeOut = 60
    frameFactory = FrameFactory()
    transport = StringTransport()
    clientConnectionPreface = frameFactory.clientConnectionPreface()
    connection.makeConnection(transport)
    connection.dataReceived(clientConnectionPreface)
    for _ in range(connection.timeOut + connection.abortTimeout):
        connection.dataReceived(frameFactory.buildRstStreamFrame(1).serialize())
        memoryReactor.advance(1)
    self.assertTrue(transport.disconnected)