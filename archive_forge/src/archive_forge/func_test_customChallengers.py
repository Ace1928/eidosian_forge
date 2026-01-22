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
def test_customChallengers(self):
    """
        L{imap4.IMAP4Server} accepts a L{dict} mapping challenge type
        names to L{twisted.mail.interfaces.IChallengeResponse}
        providers.
        """

    @implementer(IChallengeResponse, IUsernamePassword)
    class SPECIALAuth:

        def getChallenge(self):
            return b'SPECIAL'

        def setResponse(self, response):
            self.username, self.password = response.split(None, 1)

        def moreChallenges(self):
            return False

        def checkPassword(self, password):
            self.password = self.password
    special = SPECIALAuth()
    verifyObject(IChallengeResponse, special)
    server = imap4.IMAP4Server({b'SPECIAL': SPECIALAuth})
    server.portal = self.portal
    transport = StringTransport()
    server.makeConnection(transport)
    self.addCleanup(server.connectionLost, error.ConnectionDone('Connection done.'))
    self.assertIn(b'AUTH=SPECIAL', transport.value())
    transport.clear()
    server.dataReceived(b'001 AUTHENTICATE SPECIAL\r\n')
    self.assertIn(base64.b64encode(special.getChallenge()), transport.value())
    transport.clear()
    server.dataReceived(base64.b64encode(b'username password') + b'\r\n')
    self.assertEqual(transport.value(), b'001 OK Authentication successful\r\n')