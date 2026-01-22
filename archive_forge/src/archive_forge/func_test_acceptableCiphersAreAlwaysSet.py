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
def test_acceptableCiphersAreAlwaysSet(self):
    """
        If the user doesn't supply custom acceptable ciphers, a shipped secure
        default is used.  We can't check directly for it because the effective
        cipher string we set varies with platforms.
        """
    opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert)
    opts._contextFactory = FakeContext
    ctx = opts.getContext()
    self.assertEqual(opts._cipherString.encode('ascii'), ctx._cipherList)