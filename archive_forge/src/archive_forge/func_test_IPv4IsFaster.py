from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
def test_IPv4IsFaster(self):
    """
        The endpoint returns a connection to the IPv4 address.

        IPv4 ought to be the first attempt, since nameResolution (standing in
        for GAI here) returns it first. The IPv4 attempt succeeds, the
        connection is established, and a Deferred fires with the protocol
        constructed.
        """
    clientFactory = protocol.Factory()
    clientFactory.protocol = protocol.Protocol
    d = self.endpoint.connect(clientFactory)
    results = []
    d.addCallback(results.append)
    host, port, factory, timeout, bindAddress = self.mreactor.tcpClients[0]
    self.assertEqual(host, '1.2.3.4')
    self.assertEqual(port, 80)
    proto = factory.buildProtocol((host, port))
    fakeTransport = object()
    self.assertEqual(results, [])
    proto.makeConnection(fakeTransport)
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].factory, clientFactory)
    self.assertEqual([], self.mreactor.getDelayedCalls())