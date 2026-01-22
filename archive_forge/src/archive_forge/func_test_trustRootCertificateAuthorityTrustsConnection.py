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
def test_trustRootCertificateAuthorityTrustsConnection(self):
    """
        L{trustRootFromCertificates} called with certificate A will cause
        L{optionsForClientTLS} to accept client connections to a server with
        certificate B where B is signed by A.
        """
    caCert, serverCert = certificatesForAuthorityAndServer()
    trust = sslverify.trustRootFromCertificates([caCert])
    sProto, cProto, sWrap, cWrap, pump = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=serverCert.privateKey.original, serverCertificate=serverCert.original)
    self.assertEqual(cWrap.data, b'greetings!')
    self.assertIsNone(cWrap.lostReason)