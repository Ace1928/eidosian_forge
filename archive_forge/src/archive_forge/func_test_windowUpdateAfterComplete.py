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
def test_windowUpdateAfterComplete(self):
    """
        When a WindowUpdate frame is received for a stream that has been
        completed it is ignored.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
    requestBytes = f.clientConnectionPreface()
    requestBytes += b''.join((f.serialize() for f in frames))
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)

    def update_window(*args):
        windowUpdateFrame = f.buildWindowUpdateFrame(streamID=1, increment=5)
        a.dataReceived(windowUpdateFrame.serialize())

    def validate(*args):
        frames = framesFromBytes(b.value())
        self.assertIn('END_STREAM', frames[-1].flags)
    d = a._streamCleanupCallbacks[1].addCallback(update_window)
    return d.addCallback(validate)