import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_startingTLS(self):
    """
        Verify that starting TLS and succeeding at handshaking sends all the
        notifications to all the right places.
        """
    cli, svr, p = connectedServerAndClient(ServerClass=SecurableProto, ClientClass=SecurableProto)
    okc = OKCert()
    svr.certFactory = lambda: okc
    cli.callRemote(amp.StartTLS, tls_localCertificate=okc, tls_verifyAuthorities=[PretendRemoteCertificateAuthority()])
    L = []
    cli.callRemote(SecuredPing).addCallback(L.append)
    p.flush()
    self.assertEqual(okc.verifyCount, 2)
    L = []
    cli.callRemote(SecuredPing).addCallback(L.append)
    p.flush()
    self.assertEqual(L[0], {'pinged': True})