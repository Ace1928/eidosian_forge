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
def test_clientPresentsCertificate(self):
    """
        When the server verifies and the client presents a valid certificate
        for that verification by passing it to
        L{sslverify.optionsForClientTLS}, communication proceeds.
        """
    cProto, sProto, cWrapped, sWrapped, pump = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', validCertificate=True, serverVerifies=True, clientPresentsCertificate=True)
    self.assertEqual(cWrapped.data, b'greetings!')
    cErr = cWrapped.lostReason
    sErr = sWrapped.lostReason
    self.assertIsNone(cErr)
    self.assertIsNone(sErr)