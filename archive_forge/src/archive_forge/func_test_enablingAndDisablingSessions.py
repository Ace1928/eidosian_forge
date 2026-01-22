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
def test_enablingAndDisablingSessions(self):
    """
        The enableSessions argument sets the session cache mode; it defaults to
        False (at least until https://twistedmatrix.com/trac/ticket/9764 can be
        resolved).
        """
    options = sslverify.OpenSSLCertificateOptions()
    self.assertEqual(options.enableSessions, False)
    ctx = options.getContext()
    self.assertEqual(ctx.get_session_cache_mode(), SSL.SESS_CACHE_OFF)
    options = sslverify.OpenSSLCertificateOptions(enableSessions=True)
    self.assertEqual(options.enableSessions, True)
    ctx = options.getContext()
    self.assertEqual(ctx.get_session_cache_mode(), SSL.SESS_CACHE_SERVER)