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
def test_caCertsPlatformDefaults(self):
    """
        Specifying a C{trustRoot} of L{sslverify.OpenSSLDefaultPaths} when
        initializing L{sslverify.OpenSSLCertificateOptions} loads the
        platform-provided trusted certificates via C{set_default_verify_paths}.
        """
    opts = sslverify.OpenSSLCertificateOptions(trustRoot=sslverify.OpenSSLDefaultPaths())
    fc = FakeContext(SSL.TLSv1_METHOD)
    opts._contextFactory = lambda method: fc
    opts.getContext()
    self.assertTrue(fc._defaultVerifyPathsSet)