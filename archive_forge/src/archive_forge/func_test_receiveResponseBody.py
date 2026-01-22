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
def test_receiveResponseBody(self):
    """
        The C{deliverBody} method of the response object with which the
        L{Deferred} returned by L{HTTP11ClientProtocol.request} fires can be
        used to get the body of the response.
        """
    protocol = AccumulatingProtocol()
    whenFinished = protocol.closedDeferred = Deferred()
    requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
    self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r')
    result = []
    requestDeferred.addCallback(result.append)
    self.assertEqual(result, [])
    self.protocol.dataReceived(b'\n')
    response = result[0]
    response.deliverBody(protocol)
    self.protocol.dataReceived(b'foo')
    self.protocol.dataReceived(b'bar')

    def cbAllResponse(ignored):
        self.assertEqual(protocol.data, b'foobar')
        protocol.closedReason.trap(ResponseDone)
    whenFinished.addCallback(cbAllResponse)
    return whenFinished