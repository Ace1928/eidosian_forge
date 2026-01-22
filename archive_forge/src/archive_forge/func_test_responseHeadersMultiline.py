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
def test_responseHeadersMultiline(self):
    """
        The multi-line response headers are folded and added to the response
        object's C{headers} L{Headers} instance.
        """
    protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda rest: None)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
    protocol.dataReceived(b'X-Multiline: a\r\n')
    protocol.dataReceived(b'    b\r\n')
    protocol.dataReceived(b'\r\n')
    self.assertEqual(protocol.connHeaders, Headers({}))
    self.assertEqual(protocol.response.headers, Headers({b'x-multiline': [b'a    b']}))
    self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)