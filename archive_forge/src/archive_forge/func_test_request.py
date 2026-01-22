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
def test_request(self):
    """
        L{Agent.request} establishes a new connection to the host indicated by
        the host part of the URI passed to it and issues a request using the
        method, the path portion of the URI, the headers, and the body producer
        passed to it.  It returns a L{Deferred} which fires with an
        L{IResponse} from the server.
        """
    self.agent._getEndpoint = lambda *args: self
    headers = http_headers.Headers({b'foo': [b'bar']})
    body = object()
    self.agent.request(b'GET', b'http://example.com:1234/foo?bar', headers, body)
    protocol = self.protocol
    self.assertEqual(len(protocol.requests), 1)
    req, res = protocol.requests.pop()
    self.assertIsInstance(req, Request)
    self.assertEqual(req.method, b'GET')
    self.assertEqual(req.uri, b'/foo?bar')
    self.assertEqual(req.headers, http_headers.Headers({b'foo': [b'bar'], b'host': [b'example.com:1234']}))
    self.assertIdentical(req.bodyProducer, body)