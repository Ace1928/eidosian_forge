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
def test_putStartsTimeout(self):
    """
        If a connection is put back to the pool, a 240-sec timeout is started.

        When the timeout hits, the connection is closed and removed from the
        pool.
        """
    protocol = StubHTTPProtocol()
    protocol.makeConnection(StringTransport())
    self.pool._putConnection(('http', b'example.com', 80), protocol)
    self.assertEqual(protocol.transport.disconnecting, False)
    self.assertIn(protocol, self.pool._connections['http', b'example.com', 80])
    self.fakeReactor.advance(239)
    self.assertEqual(protocol.transport.disconnecting, False)
    self.assertIn(protocol, self.pool._connections['http', b'example.com', 80])
    self.assertIn(protocol, self.pool._timeouts)
    self.fakeReactor.advance(1.1)
    self.assertEqual(protocol.transport.disconnecting, True)
    self.assertNotIn(protocol, self.pool._connections['http', b'example.com', 80])
    self.assertNotIn(protocol, self.pool._timeouts)