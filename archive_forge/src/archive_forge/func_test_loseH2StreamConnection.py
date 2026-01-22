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
def test_loseH2StreamConnection(self):
    """
        Calling L{Request.loseConnection} causes all data that has previously
        been sent to be flushed, and then the stream cleanly closed.
        """
    connection = H2Connection()
    connection.requestFactory = DummyProducerHandlerProxy
    _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    stream = connection.streams[1]
    request = stream._request.original
    dataChunks = [b'hello', b'world', b'here', b'are', b'some', b'writes']
    for chunk in dataChunks:
        request.write(chunk)
    request.loseConnection()

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 9)
        self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
        self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
        self.assertTrue('END_STREAM' in frames[-1].flags)
        receivedDataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertEqual(receivedDataChunks, dataChunks + [b''])
    return connection._streamCleanupCallbacks[1].addCallback(validate)