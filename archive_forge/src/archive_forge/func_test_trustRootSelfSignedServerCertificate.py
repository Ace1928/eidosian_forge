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
def test_trustRootSelfSignedServerCertificate(self):
    """
        L{trustRootFromCertificates} called with a single self-signed
        certificate will cause L{optionsForClientTLS} to accept client
        connections to a server with that certificate.
        """
    key, cert = makeCertificate(O=b'Server Test Certificate', CN=b'server')
    selfSigned = sslverify.PrivateCertificate.fromCertificateAndKeyPair(sslverify.Certificate(cert), sslverify.KeyPair(key))
    trust = sslverify.trustRootFromCertificates([selfSigned])
    sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=selfSigned.privateKey.original, serverCertificate=selfSigned.original)
    self.assertEqual(cWrap.data, b'greetings!')
    self.assertIsNone(cWrap.lostReason)