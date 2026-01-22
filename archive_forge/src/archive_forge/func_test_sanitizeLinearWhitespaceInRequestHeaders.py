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
def test_sanitizeLinearWhitespaceInRequestHeaders(self):
    """
        Linear whitespace in request headers is replaced with a single
        space.
        """
    for component in bytesLinearWhitespaceComponents:
        headers = Headers({component: [component], b'host': [b'example.invalid']})
        transport = StringTransport()
        Request(b'GET', b'/foo', headers, None).writeTo(transport)
        lines = transport.value().split(b'\r\n')
        self.assertEqual(lines[0], b'GET /foo HTTP/1.1')
        self.assertEqual(lines[-2:], [b'', b''])
        del lines[0], lines[-2:]
        lines.remove(b'Connection: close')
        lines.remove(b'Host: example.invalid')
        sanitizedHeaderLine = b': '.join([sanitizedBytes, sanitizedBytes])
        self.assertEqual(lines, [sanitizedHeaderLine])