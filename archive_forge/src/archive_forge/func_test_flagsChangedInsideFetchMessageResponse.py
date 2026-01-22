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
def test_flagsChangedInsideFetchMessageResponse(self):
    """
        Any unrequested flag information received along with other requested
        information in an untagged I{FETCH} received in response to a request
        issued with L{IMAP4Client.fetchMessage} is passed to the
        C{flagsChanged} callback.
        """
    transport = StringTransport()
    c = StillSimplerClient()
    c.makeConnection(transport)
    c.lineReceived(b'* OK [IMAP4rev1]')

    def login():
        d = c.login(b'blah', b'blah')
        c.dataReceived(b'0001 OK LOGIN\r\n')
        return d

    def select():
        d = c.select('inbox')
        c.lineReceived(b'0002 OK SELECT')
        return d

    def fetch():
        d = c.fetchMessage('1:*')
        c.dataReceived(b'* 1 FETCH (RFC822 {24}\r\n')
        c.dataReceived(b'Subject: first subject\r\n')
        c.dataReceived(b' FLAGS (\\Seen))\r\n')
        c.dataReceived(b'* 2 FETCH (FLAGS (\\Recent \\Seen) RFC822 {25}\r\n')
        c.dataReceived(b'Subject: second subject\r\n')
        c.dataReceived(b')\r\n')
        c.dataReceived(b'0003 OK FETCH completed\r\n')
        return d

    def test(res):
        self.assertEqual(transport.value().splitlines()[-1], b'0003 FETCH 1:* (RFC822)')
        self.assertEqual(res, {1: {'RFC822': 'Subject: first subject\r\n'}, 2: {'RFC822': 'Subject: second subject\r\n'}})
        self.assertEqual(c.flags, {1: ['\\Seen'], 2: ['\\Recent', '\\Seen']})
    return login().addCallback(strip(select)).addCallback(strip(fetch)).addCallback(test)