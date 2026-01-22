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
def test_handlesPullProducer(self):
    """
        L{Request} objects that have registered pull producers get blocked and
        unblocked according to HTTP/2 flow control.
        """
    connection = H2Connection()
    connection.requestFactory = DummyPullProducerHandlerProxy
    _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    stream = connection.streams[1]
    request = stream._request.original
    producerComplete = request._actualProducer.result
    producerComplete.addCallback(lambda x: request.finish())

    def validate(streamID):
        frames = framesFromBytes(transport.value())
        self.assertTrue('END_STREAM' in frames[-1].flags)
        dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertEqual(dataChunks, [b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b''])
    return connection._streamCleanupCallbacks[1].addCallback(validate)