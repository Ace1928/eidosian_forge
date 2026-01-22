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
def test_onlyLogFailedAddresses(self):
    """
        L{smtp.SenderMixin.sentMail} adds only the addresses with failing
        SMTP response codes to the log passed to the factory's errback.
        """
    onDone = self.assertFailure(defer.Deferred(), smtp.SMTPDeliveryError)
    onDone.addCallback(lambda e: self.assertEqual(e.log, b'bob@example.com: 199 Error in sending.\n'))
    clientFactory = smtp.SMTPSenderFactory('source@address', 'recipient@address', BytesIO(b'Message body'), onDone, retries=0, timeout=0.5)
    client = clientFactory.buildProtocol(address.IPv4Address('TCP', 'example.net', 25))
    addresses = [(b'alice@example.com', 200, b'No errors here!'), (b'bob@example.com', 199, b'Error in sending.')]
    client.sentMail(199, b'Test response', 1, addresses, client.log)
    return onDone