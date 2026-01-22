from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_authenticationChallengeDecodingException(self):
    """
        When decoding a base64 encoded authentication message from the server,
        decoding errors are logged and then the client closes the connection.
        """
    transport = StringTransportWithDisconnection()
    protocol = imap4.IMAP4Client()
    transport.protocol = protocol
    protocol.makeConnection(transport)
    protocol.lineReceived(b'* OK [CAPABILITY IMAP4rev1 IDLE NAMESPACE AUTH=CRAM-MD5] Twisted IMAP4rev1 Ready')
    cAuth = imap4.CramMD5ClientAuthenticator(b'testuser')
    protocol.registerAuthenticator(cAuth)
    d = protocol.authenticate('secret')
    self.assertFailure(d, error.ConnectionDone)
    protocol.dataReceived(b'+ Something bad! and bad\r\n')
    logged = self.flushLoggedErrors(imap4.IllegalServerResponse)
    self.assertEqual(len(logged), 1)
    self.assertEqual(logged[0].value.args[0], b'Something bad! and bad')
    return d