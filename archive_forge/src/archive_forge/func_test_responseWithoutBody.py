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
def test_responseWithoutBody(self):
    """
        We safely handle responses without bodies.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyProducerHandlerProxy
    requestBytes = f.clientConnectionPreface()
    requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)
    stream = a.streams[1]
    request = stream._request.original
    cleanupCallback = a._streamCleanupCallbacks[1]
    request.unregisterProducer()
    request.finish()
    self.assertTrue(request.finished)

    def validate(streamID):
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 3)
        self.assertTrue('END_STREAM' in frames[-1].flags)
        dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertEqual(dataChunks, [b''])
    return cleanupCallback.addCallback(validate)