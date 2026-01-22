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
def test_sendNonFatalError(self):
    """
        If L{smtp.SMTPClient.sendError} is called with an L{SMTPClientError}
        which is not fatal, it sends C{"QUIT"} and waits for the server to
        close the connection.
        """
    client = smtp.SMTPClient(None)
    transport = StringTransport()
    client.makeConnection(transport)
    client.sendError(smtp.SMTPClientError(123, 'foo', isFatal=False))
    self.assertEqual(transport.value(), b'QUIT\r\n')
    self.assertFalse(transport.disconnecting)