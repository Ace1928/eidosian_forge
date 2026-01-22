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
def test_protocolErrorTerminatesConnection(self):
    """
        A protocol error from the remote peer terminates the connection.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    requestBytes = f.clientConnectionPreface()
    requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
    requestBytes += f.buildPushPromiseFrame(streamID=1, promisedStreamID=2, headers=self.getRequestHeaders, flags=['END_HEADERS']).serialize()
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)
        if b.disconnecting:
            break
    frames = framesFromBytes(b.value())
    self.assertEqual(len(frames), 3)
    self.assertTrue(isinstance(frames[-1], hyperframe.frame.GoAwayFrame))
    self.assertTrue(b.disconnecting)