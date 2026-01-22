from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def test_bufferEarlyData(self):
    """
        If data is delivered to the L{Response} before a protocol is registered
        with C{deliverBody}, that data is buffered until the protocol is
        registered and then is delivered.
        """
    bytes = []

    class ListConsumer(Protocol):

        def dataReceived(self, data):
            bytes.append(data)
    protocol = ListConsumer()
    response = justTransportResponse(StringTransport())
    response._bodyDataReceived(b'foo')
    response._bodyDataReceived(b'bar')
    response.deliverBody(protocol)
    response._bodyDataReceived(b'baz')
    self.assertEqual(bytes, [b'foo', b'bar', b'baz'])
    self.assertIdentical(response._bodyBuffer, None)