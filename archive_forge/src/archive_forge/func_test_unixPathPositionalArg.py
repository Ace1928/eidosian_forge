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
def test_unixPathPositionalArg(self):
    """
        When passed a UNIX strports description specifying path as a positional
        argument, L{endpoints.clientFromString} returns a L{UNIXClientEndpoint}
        instance initialized with the values from the string.
        """
    reactor = object()
    client = endpoints.clientFromString(reactor, 'unix:/var/foo/bar:lockfile=1:timeout=9')
    self.assertIsInstance(client, endpoints.UNIXClientEndpoint)
    self.assertIs(client._reactor, reactor)
    self.assertEqual(client._path, '/var/foo/bar')
    self.assertEqual(client._timeout, 9)
    self.assertTrue(client._checkPID)