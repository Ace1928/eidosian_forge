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
def test_trustRootFromCertificatesUntrusted(self):
    """
        L{trustRootFromCertificates} called with certificate A will cause
        L{optionsForClientTLS} to disallow any connections to a server with
        certificate B where B is not signed by A.
        """
    key, cert = makeCertificate(O=b'Server Test Certificate', CN=b'server')
    serverCert = sslverify.PrivateCertificate.fromCertificateAndKeyPair(sslverify.Certificate(cert), sslverify.KeyPair(key))
    untrustedCert = sslverify.Certificate(makeCertificate(O=b'CA Test Certificate', CN=b'unknown CA')[1])
    trust = sslverify.trustRootFromCertificates([untrustedCert])
    sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=serverCert.privateKey.original, serverCertificate=serverCert.original)
    self.assertEqual(cWrap.data, b'')
    self.assertEqual(cWrap.lostReason.type, SSL.Error)
    err = cWrap.lostReason.value
    self.assertEqual(err.args[0][0][2], 'tlsv1 alert unknown ca')