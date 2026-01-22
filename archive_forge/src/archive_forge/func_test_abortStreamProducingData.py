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
def test_abortStreamProducingData(self):
    """
        The H2Stream data implements IPushProducer, and can have its data
        production controlled by the Request if the Request chooses to.
        When the production is stopped, that causes the stream connection to
        be lost.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = AbortingConsumerDummyHandlerProxy
    frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
    frames[-1].flags = set()
    requestBytes = f.clientConnectionPreface()
    requestBytes += b''.join((f.serialize() for f in frames))
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)
    request = a.streams[1]._request.original
    self.assertFalse(request._requestReceived)
    cleanupCallback = a._streamCleanupCallbacks[1]
    request.acceptData()

    def validate(streamID):
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 2)
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.RstStreamFrame))
        self.assertEqual(frames[-1].stream_id, 1)
    return cleanupCallback.addCallback(validate)