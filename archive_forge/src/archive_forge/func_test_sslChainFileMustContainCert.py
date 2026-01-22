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
@skipIf(skipSSL, skipSSLReason)
def test_sslChainFileMustContainCert(self):
    """
        If C{extraCertChain} is passed, it has to contain at least one valid
        certificate in PEM format.
        """
    fp = FilePath(self.mktemp())
    fp.create().close()
    with self.assertRaises(ValueError) as caught:
        endpoints.serverFromString(object(), self.SSL_CHAIN_TEMPLATE % (escapedPEMPathName, endpoints.quoteStringArgument(fp.path)))
    self.assertEqual(str(caught.exception), "Specified chain file '%s' doesn't contain any valid certificates in PEM format." % (fp.path,))