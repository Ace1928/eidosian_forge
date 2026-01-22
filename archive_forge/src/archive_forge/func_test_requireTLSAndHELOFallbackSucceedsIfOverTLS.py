import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
def test_requireTLSAndHELOFallbackSucceedsIfOverTLS(self):
    """
        If TLS is provided at the transport level, we can honour the HELO
        fallback if we're set to require TLS.
        """
    transport = StringTransport()
    directlyProvides(transport, interfaces.ISSLTransport)
    self.clientProtocol.requireAuthentication = False
    self.clientProtocol.requireTransportSecurity = True
    self.clientProtocol.heloFallback = True
    self.clientProtocol.makeConnection(transport)
    self.clientProtocol.dataReceived(b'220 localhost\r\n')
    transport.clear()
    self.clientProtocol.dataReceived(b'500 not an esmtp server\r\n')
    self.assertEqual(b'HELO testuser\r\n', transport.value())