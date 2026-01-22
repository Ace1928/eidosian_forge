from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_getPeerCertificate(self):
    """
        L{TLSMemoryBIOProtocol.getPeerCertificate} returns the
        L{OpenSSL.crypto.X509} instance representing the peer's
        certificate.
        """
    clientFactory = ClientFactory()
    clientFactory.protocol = Protocol
    clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
    wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
    sslClientProtocol = wrapperFactory.buildProtocol(None)
    serverFactory = ServerFactory()
    serverFactory.protocol = Protocol
    serverContextFactory = ServerTLSContext()
    wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
    sslServerProtocol = wrapperFactory.buildProtocol(None)
    loopbackAsync(sslServerProtocol, sslClientProtocol)

    def cbHandshook(ignored):
        cert = sslClientProtocol.getPeerCertificate()
        self.assertIsInstance(cert, crypto.X509)
        self.assertEqual(cert.digest('sha256'), b'D6:F2:2C:74:3B:E2:5E:F9:CA:DA:47:08:14:78:20:75:78:95:9E:52:BD:D2:7C:77:DD:D4:EE:DE:33:BF:34:40')
    handshakeDeferred.addCallback(cbHandshook)
    return handshakeDeferred