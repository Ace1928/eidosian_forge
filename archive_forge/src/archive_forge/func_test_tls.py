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
def test_tls(self):
    """
        When passed a string endpoint description beginning with C{tls:},
        L{clientFromString} returns a client endpoint initialized with the
        values from the string.
        """
    reactor = MemoryReactor()
    endpoint = endpoints.clientFromString(deterministicResolvingReactor(reactor, ['127.0.0.1']), 'tls:localhost:4321:privateKey={}:certificate={}:trustRoots={}'.format(escapedPEMPathName, escapedPEMPathName, endpoints.quoteStringArgument(pemPath.parent().path)).encode('ascii'))
    d = endpoint.connect(Factory.forProtocol(Protocol))
    host, port, factory, timeout, bindAddress = reactor.tcpClients.pop()
    clientProtocol = factory.buildProtocol(None)
    self.assertNoResult(d)
    assert clientProtocol is not None
    serverCert = PrivateCertificate.loadPEM(pemPath.getContent())
    serverOptions = CertificateOptions(privateKey=serverCert.privateKey.original, certificate=serverCert.original, extraCertChain=[Certificate.loadPEM(chainPath.getContent()).original], trustRoot=serverCert)
    plainServer = Protocol()
    serverProtocol = TLSMemoryBIOFactory(serverOptions, isClient=False, wrappedFactory=Factory.forProtocol(lambda: plainServer)).buildProtocol(None)
    sProto, cProto, pump = connectedServerAndClient(lambda: serverProtocol, lambda: clientProtocol)
    plainServer.transport.write(b'hello\r\n')
    plainClient = self.successResultOf(d)
    plainClient.transport.write(b'hi you too\r\n')
    pump.flush()
    self.assertFalse(plainServer.transport.disconnecting)
    self.assertFalse(plainClient.transport.disconnecting)
    self.assertFalse(plainServer.transport.disconnected)
    self.assertFalse(plainClient.transport.disconnected)
    peerCertificate = Certificate.peerFromTransport(plainServer.transport)
    self.assertEqual(peerCertificate, Certificate.loadPEM(pemPath.getContent()))