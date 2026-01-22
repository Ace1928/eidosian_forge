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
def test_verification(self):
    """
        Check certificates verification building custom certificates data.
        """
    clientDN = sslverify.DistinguishedName(commonName='client')
    clientKey = sslverify.KeyPair.generate()
    clientCertReq = clientKey.certificateRequest(clientDN)
    serverDN = sslverify.DistinguishedName(commonName='server')
    serverKey = sslverify.KeyPair.generate()
    serverCertReq = serverKey.certificateRequest(serverDN)
    clientSelfCertReq = clientKey.certificateRequest(clientDN)
    clientSelfCertData = clientKey.signCertificateRequest(clientDN, clientSelfCertReq, lambda dn: True, 132)
    clientSelfCert = clientKey.newCertificate(clientSelfCertData)
    serverSelfCertReq = serverKey.certificateRequest(serverDN)
    serverSelfCertData = serverKey.signCertificateRequest(serverDN, serverSelfCertReq, lambda dn: True, 516)
    serverSelfCert = serverKey.newCertificate(serverSelfCertData)
    clientCertData = serverKey.signCertificateRequest(serverDN, clientCertReq, lambda dn: True, 7)
    clientCert = clientKey.newCertificate(clientCertData)
    serverCertData = clientKey.signCertificateRequest(clientDN, serverCertReq, lambda dn: True, 42)
    serverCert = serverKey.newCertificate(serverCertData)
    onData = defer.Deferred()
    serverOpts = serverCert.options(serverSelfCert)
    clientOpts = clientCert.options(clientSelfCert)
    self.loopback(serverOpts, clientOpts, onData=onData)
    return onData.addCallback(lambda result: self.assertEqual(result, WritingProtocol.byte))