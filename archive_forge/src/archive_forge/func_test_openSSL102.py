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
def test_openSSL102(self):
    """
        OpenSSL 1.0.2 does not set ECDH curves by default, but
        C{SSL_CTX_set_ecdh_auto} requests that a context choose a
        secure set curves automatically.
        """
    context = SSL.Context(SSL.SSLv23_METHOD)
    chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_102, openSSLlib=self.lib, openSSLcrypto=self.crypto)
    chooser.configureECDHCurve(context)
    self.assertEqual(self.libState.ecdhContexts, [context._context])
    self.assertEqual(self.libState.ecdhValues, [True])
    self.assertFalse(self.cryptoState.getEllipticCurveCalls)
    self.assertIsNone(self.context._ecCurve)