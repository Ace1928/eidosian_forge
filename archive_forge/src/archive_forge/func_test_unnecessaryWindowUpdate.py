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
def test_unnecessaryWindowUpdate(self):
    """
        When a WindowUpdate frame is received for the whole connection but no
        data is currently waiting, nothing exciting happens.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
    frames.insert(1, f.buildWindowUpdateFrame(streamID=0, increment=5))
    requestBytes = f.clientConnectionPreface()
    requestBytes += b''.join((f.serialize() for f in frames))
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)

    def validate(streamID):
        frames = framesFromBytes(b.value())
        self.assertTrue('END_STREAM' in frames[-1].flags)
        actualResponseData = b''.join((f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)))
        self.assertEqual(self.postResponseData, actualResponseData)
    return a._streamCleanupCallbacks[1].addCallback(validate)