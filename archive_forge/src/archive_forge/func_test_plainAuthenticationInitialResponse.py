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
def test_plainAuthenticationInitialResponse(self):
    """
        The response to the first challenge may be included on the AUTH command
        line.  Test that this is also supported.
        """
    loginArgs = []
    self.server.portal = self.portalFactory(loginArgs)
    self.server.dataReceived(b'EHLO\r\n')
    self.transport.clear()
    self.assertServerResponse(b'AUTH LOGIN ' + base64.b64encode(b'username').strip() + b'\r\n', [b'334 ' + base64.b64encode(b'Password\x00').strip()])
    self.assertServerResponse(base64.b64encode(b'password').strip() + b'\r\n', [])
    self.assertServerAuthenticated(loginArgs)