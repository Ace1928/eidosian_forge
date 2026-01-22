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
def test_delayWrites(self):
    """
        Delaying writes from L{Request} causes the L{H2Connection} to block on
        sending until data is available. However, data is *not* sent if there's
        no room in the flow control window.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DelayedHTTPHandlerProxy
    requestBytes = f.clientConnectionPreface()
    requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
    requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)
    stream = a.streams[1]
    request = stream._request.original
    request.write(b'fiver')
    dataChunks = [b'here', b'are', b'some', b'writes']

    def write_chunks():
        for chunk in dataChunks:
            request.write(chunk)
        request.finish()
    d = task.deferLater(reactor, 0.01, write_chunks)
    d.addCallback(lambda *args: a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize()))

    def validate(streamID):
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 9)
        self.assertTrue(all((f.stream_id == 1 for f in frames[2:])))
        self.assertTrue(isinstance(frames[2], hyperframe.frame.HeadersFrame))
        self.assertTrue('END_STREAM' in frames[-1].flags)
        receivedDataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertEqual(receivedDataChunks, [b'fiver'] + dataChunks + [b''])
    return a._streamCleanupCallbacks[1].addCallback(validate)