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
def test_otherConnectionsCancelled(self):
    """
        Once the endpoint returns a successful connection, all the other
        pending connections are cancelled.

        Here, the second connection attempt, i.e. IPv6, succeeds, and the
        pending first attempt, i.e. IPv4, is cancelled.
        """
    clientFactory = protocol.Factory()
    clientFactory.protocol = protocol.Protocol
    d = self.endpoint.connect(clientFactory)
    results = []
    d.addCallback(results.append)
    self.mreactor.advance(0.3)
    host, port, factory, timeout, bindAddress = self.mreactor.tcpClients[1]
    proto = factory.buildProtocol((host, port))
    fakeTransport = object()
    proto.makeConnection(fakeTransport)
    self.assertEqual(True, self.mreactor.tcpClients[0][2]._connector.stoppedConnecting)
    self.assertEqual([], self.mreactor.getDelayedCalls())