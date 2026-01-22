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
def test_returnsTupleOfICiphers(self):
    """
        L{sslverify._expandCipherString} always returns a L{tuple} of
        L{interfaces.ICipher}.
        """
    ciphers = sslverify._expandCipherString('ALL', SSL.SSLv23_METHOD, 0)
    self.assertIsInstance(ciphers, tuple)
    bogus = []
    for c in ciphers:
        if not interfaces.ICipher.providedBy(c):
            bogus.append(c)
    self.assertEqual([], bogus)