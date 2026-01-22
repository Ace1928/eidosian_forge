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
def test_abortRequestWithCircuitBreaker(self):
    """
        Aborting a request associated with a paused connection that's
        reached its buffered control frame limit causes that
        connection to be aborted.
        """
    memoryReactor = MemoryReactorClock()
    connection = H2Connection(memoryReactor)
    connection.callLater = memoryReactor.callLater
    connection.requestFactory = DummyHTTPHandlerProxy
    frameFactory = FrameFactory()
    transport = StringTransport()
    clientConnectionPreface = frameFactory.clientConnectionPreface()
    connection.makeConnection(transport)
    connection.dataReceived(clientConnectionPreface)
    streamID = 1
    headersFrameData = frameFactory.buildHeadersFrame(headers=self.postRequestHeaders, streamID=streamID).serialize()
    connection.dataReceived(headersFrameData)
    connection.pauseProducing()
    connection._maxBufferedControlFrameBytes = 0
    transport.clear()
    connection.abortRequest(streamID)
    self.assertFalse(transport.value())
    self.assertTrue(transport.disconnected)