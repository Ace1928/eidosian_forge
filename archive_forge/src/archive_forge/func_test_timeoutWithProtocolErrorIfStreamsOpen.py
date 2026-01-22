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
def test_timeoutWithProtocolErrorIfStreamsOpen(self):
    """
        When a L{H2Connection} times out with active streams, the error code
        returned is L{h2.errors.ErrorCodes.PROTOCOL_ERROR}.
        """
    frameFactory = FrameFactory()
    frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
    initialData = frameFactory.clientConnectionPreface()
    initialData += b''.join((f.serialize() for f in frames))
    reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyProducerHandler)
    reactor.advance(101)
    self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.PROTOCOL_ERROR, lastStreamID=1)
    self.assertTrue(transport.disconnecting)