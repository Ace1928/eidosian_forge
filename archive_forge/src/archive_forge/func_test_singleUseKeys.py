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
def test_singleUseKeys(self):
    """
        If C{singleUseKeys} is set, every context must have
        C{OP_SINGLE_DH_USE} and C{OP_SINGLE_ECDH_USE} set.
        """
    opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, enableSingleUseKeys=True)
    opts._contextFactory = FakeContext
    ctx = opts.getContext()
    options = SSL.OP_SINGLE_DH_USE | SSL.OP_SINGLE_ECDH_USE
    self.assertEqual(options, ctx._options & options)