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
def test_tlsProtocolsNoMethodWithAtLeast(self):
    """
        Passing C{raiseMinimumTo} along with C{method} to
        L{sslverify.OpenSSLCertificateOptions} will cause it to raise an
        exception.
        """
    with self.assertRaises(TypeError) as e:
        sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.SSLv23_METHOD, raiseMinimumTo=sslverify.TLSVersion.TLSv1_2)
    self.assertIn('method', e.exception.args[0])
    self.assertIn('raiseMinimumTo', e.exception.args[0])
    self.assertIn('exclusive', e.exception.args[0])