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
def test_sendChunkedRequestBody(self):
    """
        L{Request.writeTo} uses chunked encoding to write data from the request
        body producer to the given transport.  It registers the request body
        producer with the transport.
        """
    producer = StringProducer(UNKNOWN_LENGTH)
    request = Request(b'POST', b'/bar', _boringHeaders, producer)
    request.writeTo(self.transport)
    self.assertNotIdentical(producer.consumer, None)
    self.assertIdentical(self.transport.producer, producer)
    self.assertTrue(self.transport.streaming)
    self.assertEqual(self.transport.value(), b'POST /bar HTTP/1.1\r\nConnection: close\r\nTransfer-Encoding: chunked\r\nHost: example.com\r\n\r\n')
    self.transport.clear()
    producer.consumer.write(b'x' * 3)
    producer.consumer.write(b'y' * 15)
    producer.finished.callback(None)
    self.assertIdentical(self.transport.producer, None)
    self.assertEqual(self.transport.value(), b'3\r\nxxx\r\nf\r\nyyyyyyyyyyyyyyy\r\n0\r\n\r\n')