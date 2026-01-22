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
def test_circuitBreakerAbortsAfterProtocolError(self):
    """
        A client that triggers a L{h2.exceptions.ProtocolError} over a
        paused connection that's reached its buffered control frame
        limit causes that connection to be aborted.
        """
    memoryReactor = MemoryReactorClock()
    connection = H2Connection(memoryReactor)
    connection.callLater = memoryReactor.callLater
    frameFactory = FrameFactory()
    transport = StringTransport()
    clientConnectionPreface = frameFactory.clientConnectionPreface()
    connection.makeConnection(transport)
    connection.dataReceived(clientConnectionPreface)
    connection.pauseProducing()
    connection._maxBufferedControlFrameBytes = 0
    invalidData = frameFactory.buildDataFrame(data=b'yo', streamID=240).serialize()
    connection.dataReceived(invalidData)
    self.assertTrue(transport.disconnected)