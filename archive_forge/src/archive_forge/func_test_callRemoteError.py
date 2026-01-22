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
def test_callRemoteError(self):
    """
        Check that callRemote raises an exception when called with a
        L{amp.StartTLS}.
        """
    cli, svr, p = connectedServerAndClient(ServerClass=SecurableProto, ClientClass=SecurableProto)
    okc = OKCert()
    svr.certFactory = lambda: okc
    return self.assertFailure(cli.callRemote(amp.StartTLS, tls_localCertificate=okc, tls_verifyAuthorities=[PretendRemoteCertificateAuthority()]), RuntimeError)