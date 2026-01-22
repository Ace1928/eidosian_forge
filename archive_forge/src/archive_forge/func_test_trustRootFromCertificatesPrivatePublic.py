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
def test_trustRootFromCertificatesPrivatePublic(self):
    """
        L{trustRootFromCertificates} accepts either a L{sslverify.Certificate}
        or a L{sslverify.PrivateCertificate} instance.
        """
    privateCert = sslverify.PrivateCertificate.loadPEM(A_KEYPAIR)
    cert = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
    mt = sslverify.trustRootFromCertificates([privateCert, cert])
    sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=mt, privateKey=privateCert.privateKey.original, serverCertificate=privateCert.original)
    self.assertEqual(cWrap.data, b'greetings!')
    self.assertIsNone(cWrap.lostReason)