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
def test_SSL_CTX_set_ecdh_auto(self):
    """
        L{FakeLib.SSL_CTX_set_ecdh_auto} records context and value it
        was called with.
        """
    state = FakeLibState(setECDHAutoRaises=None)
    lib = FakeLib(state)
    self.assertNot(state.ecdhContexts)
    self.assertNot(state.ecdhValues)
    context, value = ('CONTEXT', True)
    lib.SSL_CTX_set_ecdh_auto(context, value)
    self.assertEqual(state.ecdhContexts, [context])
    self.assertEqual(state.ecdhValues, [True])