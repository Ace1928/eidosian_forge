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
def test_retryWithNewConnection(self):
    """
        L{client.HTTPConnectionPool} creates
        {client._RetryingHTTP11ClientProtocol} with a new connection factory
        method that creates a new connection using the same key and endpoint
        as the wrapped connection.
        """
    pool = client.HTTPConnectionPool(Clock())
    key = 123
    endpoint = DummyEndpoint()
    newConnections = []

    def newConnection(k, e):
        newConnections.append((k, e))
    pool._newConnection = newConnection
    protocol = StubHTTPProtocol()
    protocol.makeConnection(StringTransport())
    pool._putConnection(key, protocol)
    d = pool.getConnection(key, endpoint)

    def gotConnection(connection):
        self.assertIsInstance(connection, client._RetryingHTTP11ClientProtocol)
        self.assertIdentical(connection._clientProtocol, protocol)
        self.assertEqual(newConnections, [])
        connection._newConnection()
        self.assertEqual(len(newConnections), 1)
        self.assertEqual(newConnections[0][0], key)
        self.assertIdentical(newConnections[0][1], endpoint)
    return d.addCallback(gotConnection)