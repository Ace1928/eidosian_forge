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
def test_timeoutResetByResponseData(self):
    """
        When a L{H2Connection} sends data, the timeout is reset.
        """
    frameFactory = FrameFactory()
    initialData = b''
    requests = []
    frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
    initialData = frameFactory.clientConnectionPreface()
    initialData += b''.join((f.serialize() for f in frames))

    def saveRequest(stream, queued):
        req = DelayedHTTPHandler(stream, queued=queued)
        requests.append(req)
        return req
    reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=saveRequest)
    conn.dataReceived(frameFactory.clientConnectionPreface())
    reactor.advance(99)
    self.assertEquals(len(requests), 1)
    for x in range(10):
        requests[0].write(b'some bytes')
        reactor.advance(99)
        self.assertFalse(transport.disconnecting)
    reactor.advance(2)
    self.assertTimedOut(transport.value(), frameCount=13, errorCode=h2.errors.ErrorCodes.PROTOCOL_ERROR, lastStreamID=1)