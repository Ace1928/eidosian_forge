import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def serviceIdentitySetup(self, clientHostname, serverHostname, serverContextSetup=lambda ctx: None, validCertificate=True, clientPresentsCertificate=False, validClientCertificate=True, serverVerifies=False, buggyInfoCallback=False, fakePlatformTrust=False, useDefaultTrust=False):
    """
        Connect a server and a client.

        @param clientHostname: The I{client's idea} of the server's hostname;
            passed as the C{hostname} to the
            L{sslverify.OpenSSLCertificateOptions} instance.
        @type clientHostname: L{unicode}

        @param serverHostname: The I{server's own idea} of the server's
            hostname; present in the certificate presented by the server.
        @type serverHostname: L{unicode}

        @param serverContextSetup: a 1-argument callable invoked with the
            L{OpenSSL.SSL.Context} after it's produced.
        @type serverContextSetup: L{callable} taking L{OpenSSL.SSL.Context}
            returning L{None}.

        @param validCertificate: Is the server's certificate valid?  L{True} if
            so, L{False} otherwise.
        @type validCertificate: L{bool}

        @param clientPresentsCertificate: Should the client present a
            certificate to the server?  Defaults to 'no'.
        @type clientPresentsCertificate: L{bool}

        @param validClientCertificate: If the client presents a certificate,
            should it actually be a valid one, i.e. signed by the same CA that
            the server is checking?  Defaults to 'yes'.
        @type validClientCertificate: L{bool}

        @param serverVerifies: Should the server verify the client's
            certificate?  Defaults to 'no'.
        @type serverVerifies: L{bool}

        @param buggyInfoCallback: Should we patch the implementation so that
            the C{info_callback} passed to OpenSSL to have a bug and raise an
            exception (L{ZeroDivisionError})?  Defaults to 'no'.
        @type buggyInfoCallback: L{bool}

        @param fakePlatformTrust: Should we fake the platformTrust to be the
            same as our fake server certificate authority, so that we can test
            it's being used?  Defaults to 'no' and we just pass platform trust.
        @type fakePlatformTrust: L{bool}

        @param useDefaultTrust: Should we avoid passing the C{trustRoot} to
            L{ssl.optionsForClientTLS}?  Defaults to 'no'.
        @type useDefaultTrust: L{bool}

        @return: the client TLS protocol, the client wrapped protocol,
            the server TLS protocol, the server wrapped protocol and
            an L{IOPump} which, when its C{pump} and C{flush} methods are
            called, will move data between the created client and server
            protocol instances
        @rtype: 5-L{tuple} of 4 L{IProtocol}s and L{IOPump}
        """
    serverCA, serverCert = certificatesForAuthorityAndServer(serverHostname)
    other = {}
    passClientCert = None
    clientCA, clientCert = certificatesForAuthorityAndServer('client')
    if serverVerifies:
        other.update(trustRoot=clientCA)
    if clientPresentsCertificate:
        if validClientCertificate:
            passClientCert = clientCert
        else:
            bogusCA, bogus = certificatesForAuthorityAndServer('client')
            passClientCert = bogus
    serverOpts = sslverify.OpenSSLCertificateOptions(privateKey=serverCert.privateKey.original, certificate=serverCert.original, **other)
    serverContextSetup(serverOpts.getContext())
    if not validCertificate:
        serverCA, otherServer = certificatesForAuthorityAndServer(serverHostname)
    if buggyInfoCallback:

        def broken(*a, **k):
            """
                Raise an exception.

                @param a: Arguments for an C{info_callback}

                @param k: Keyword arguments for an C{info_callback}
                """
            1 / 0
        self.patch(sslverify.ClientTLSOptions, '_identityVerifyingInfoCallback', broken)
    signature = {'hostname': clientHostname}
    if passClientCert:
        signature.update(clientCertificate=passClientCert)
    if not useDefaultTrust:
        signature.update(trustRoot=serverCA)
    if fakePlatformTrust:
        self.patch(sslverify, 'platformTrust', lambda: serverCA)
    clientOpts = sslverify.optionsForClientTLS(**signature)

    class GreetingServer(protocol.Protocol):
        greeting = b'greetings!'
        lostReason = None
        data = b''

        def connectionMade(self):
            self.transport.write(self.greeting)

        def dataReceived(self, data):
            self.data += data

        def connectionLost(self, reason):
            self.lostReason = reason

    class GreetingClient(protocol.Protocol):
        greeting = b'cheerio!'
        data = b''
        lostReason = None

        def connectionMade(self):
            self.transport.write(self.greeting)

        def dataReceived(self, data):
            self.data += data

        def connectionLost(self, reason):
            self.lostReason = reason
    serverWrappedProto = GreetingServer()
    clientWrappedProto = GreetingClient()
    clientFactory = protocol.Factory()
    clientFactory.protocol = lambda: clientWrappedProto
    serverFactory = protocol.Factory()
    serverFactory.protocol = lambda: serverWrappedProto
    self.serverOpts = serverOpts
    self.clientOpts = clientOpts
    clock = Clock()
    clientTLSFactory = TLSMemoryBIOFactory(clientOpts, isClient=True, wrappedFactory=clientFactory, clock=clock)
    serverTLSFactory = TLSMemoryBIOFactory(serverOpts, isClient=False, wrappedFactory=serverFactory, clock=clock)
    cProto, sProto, pump = connectedServerAndClient(lambda: serverTLSFactory.buildProtocol(None), lambda: clientTLSFactory.buildProtocol(None), clock=clock)
    pump.flush()
    return (cProto, sProto, clientWrappedProto, serverWrappedProto, pump)