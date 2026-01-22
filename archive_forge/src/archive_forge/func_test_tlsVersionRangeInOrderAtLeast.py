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
def test_tlsVersionRangeInOrderAtLeast(self):
    """
        Passing out of order TLS versions to C{raiseMinimumTo} and
        C{lowerMaximumSecurityTo} will cause it to raise an exception.
        """
    with self.assertRaises(ValueError) as e:
        sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, raiseMinimumTo=sslverify.TLSVersion.TLSv1_0, lowerMaximumSecurityTo=sslverify.TLSVersion.SSLv3)
    self.assertEqual(e.exception.args, ('raiseMinimumTo needs to be lower than lowerMaximumSecurityTo',))