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
def test_resolveNameDelivers(self):
    """
        The resolution receiver begins, and resolved hostnames are
        delivered before it completes.
        """
    port = 80
    ipv4Host = '1.2.3.4'
    ipv6Host = '1::2::3::4'
    self.resolver.resolveHostName(self.receiver, 'example.com')
    self.fakeResolverReturns.callback([(AF_INET, SOCK_STREAM, IPPROTO_TCP, '', (ipv4Host, port)), (AF_INET6, SOCK_STREAM, IPPROTO_TCP, '', (ipv6Host, port))])
    self.assertEqual(1, len(self.resolutionBeganCalls))
    self.assertEqual(self.resolutionBeganCalls[0].name, 'example.com')
    self.assertEqual(self.addressResolvedCalls, [IPv4Address('TCP', ipv4Host, port), IPv6Address('TCP', ipv6Host, port)])
    self.assertEqual(self.resolutionCompleteCallCount, 1)