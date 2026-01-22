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
def test_timeoutConnection(self):
    """
        L{smtp.SMTPClient.timeoutConnection} calls the C{sendError} hook with a
        fatal L{SMTPTimeoutError} with the current line log.
        """
    errors = []
    client = MySMTPClient()
    client.sendError = errors.append
    client.makeConnection(StringTransport())
    client.lineReceived(b'220 hello')
    client.timeoutConnection()
    self.assertIsInstance(errors[0], smtp.SMTPTimeoutError)
    self.assertTrue(errors[0].isFatal)
    self.assertEqual(bytes(errors[0]), b'Timeout waiting for SMTP server response\n<<< 220 hello\n>>> HELO foo.baz\n')