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
def test_cacheRemovesOldest(self):
    """
        Verify that when the cache is full, and a new entry is added,
        the oldest entry is removed.
        """
    trustRoot = CustomOpenSSLTrustRoot()
    wrappedPolicy = BrowserLikePolicyForHTTPS(trustRoot=trustRoot)
    policy = HostnameCachingHTTPSPolicy(wrappedPolicy)
    for i in range(0, 20):
        hostname = 'host' + str(i)
        policy.creatorForNetloc(hostname.encode('ascii'), 8675)
    host0 = 'host0'
    policy.creatorForNetloc(host0.encode('ascii'), 309)
    self.assertIn(host0, policy._cache)
    self.assertEquals(20, len(policy._cache))
    hostn = 'new'
    policy.creatorForNetloc(hostn.encode('ascii'), 309)
    host1 = 'host1'
    self.assertNotIn(host1, policy._cache)
    self.assertEquals(20, len(policy._cache))
    self.assertIn(hostn, policy._cache)
    self.assertIn(host0, policy._cache)
    for _ in range(20):
        policy.creatorForNetloc(host0.encode('ascii'), 8675)
    hostNPlus1 = 'new1'
    policy.creatorForNetloc(hostNPlus1.encode('ascii'), 800)
    self.assertNotIn('host2', policy._cache)
    self.assertEquals(20, len(policy._cache))
    self.assertIn(hostNPlus1, policy._cache)
    self.assertIn(hostn, policy._cache)
    self.assertIn(host0, policy._cache)