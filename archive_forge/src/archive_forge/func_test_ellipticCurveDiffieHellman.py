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
def test_ellipticCurveDiffieHellman(self):
    """
        Connections use ECDH when OpenSSL supports it.
        """
    if not get_elliptic_curves():
        raise SkipTest('OpenSSL does not support ECDH.')
    onData = defer.Deferred()
    self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, requireCertificate=False, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_3), sslverify.OpenSSLCertificateOptions(requireCertificate=False, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_3), onData=onData)

    @onData.addCallback
    def assertECDH(_):
        self.assertEqual(len(self.clientConn.factory.protocols), 1)
        [clientProtocol] = self.clientConn.factory.protocols
        cipher = clientProtocol.getHandle().get_cipher_name()
        self.assertIn('ECDH', cipher)
    return onData