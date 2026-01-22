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
def test_headResponseContentLengthEntityHeader(self):
    """
        If a HEAD request is made, the I{Content-Length} header in the response
        is added to the response headers, not the connection control headers.
        """
    protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda rest: None)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
    protocol.dataReceived(b'Content-Length: 123\r\n')
    protocol.dataReceived(b'\r\n')
    self.assertEqual(protocol.response.headers, Headers({b'content-length': [b'123']}))
    self.assertEqual(protocol.connHeaders, Headers({}))
    self.assertEqual(protocol.response.length, 0)