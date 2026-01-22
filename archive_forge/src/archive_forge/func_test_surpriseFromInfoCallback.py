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
def test_surpriseFromInfoCallback(self):
    """
        pyOpenSSL isn't always so great about reporting errors.  If one occurs
        in the verification info callback, it should be logged and the
        connection should be shut down (if possible, anyway; the app_data could
        be clobbered but there's no point testing for that).
        """
    cProto, sProto, cWrapped, sWrapped, pump = self.serviceIdentitySetup('correct-host.example.com', 'correct-host.example.com', buggyInfoCallback=True)
    self.assertEqual(cWrapped.data, b'')
    self.assertEqual(sWrapped.data, b'')
    cErr = cWrapped.lostReason.value
    sErr = sWrapped.lostReason.value
    self.assertIsInstance(cErr, ZeroDivisionError)
    self.assertIsInstance(sErr, (ConnectionClosed, SSL.Error))
    errors = self.flushLoggedErrors(ZeroDivisionError)
    self.assertTrue(errors)