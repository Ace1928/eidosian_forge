from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
def test_gzipEncodingResponse(self):
    """
        If the response has a C{gzip} I{Content-Encoding} header,
        L{GzipDecoder} wraps the response to return uncompressed data to the
        user.
        """
    deferred = self.agent.request(b'GET', b'http://example.com/foo')
    req, res = self.protocol.requests.pop()
    headers = http_headers.Headers({b'foo': [b'bar'], b'content-encoding': [b'gzip']})
    transport = StringTransport()
    response = Response((b'HTTP', 1, 1), 200, b'OK', headers, transport)
    response.length = 12
    res.callback(response)
    compressor = zlib.compressobj(2, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    data = compressor.compress(b'x' * 6) + compressor.compress(b'y' * 4) + compressor.flush()

    def checkResponse(result):
        self.assertNotIdentical(result, response)
        self.assertEqual(result.version, (b'HTTP', 1, 1))
        self.assertEqual(result.code, 200)
        self.assertEqual(result.phrase, b'OK')
        self.assertEqual(list(result.headers.getAllRawHeaders()), [(b'Foo', [b'bar'])])
        self.assertEqual(result.length, UNKNOWN_LENGTH)
        self.assertRaises(AttributeError, getattr, result, 'unknown')
        response._bodyDataReceived(data[:5])
        response._bodyDataReceived(data[5:])
        response._bodyDataFinished()
        protocol = SimpleAgentProtocol()
        result.deliverBody(protocol)
        self.assertEqual(protocol.received, [b'x' * 6 + b'y' * 4])
        return defer.gatherResults([protocol.made, protocol.finished])
    deferred.addCallback(checkResponse)
    return deferred