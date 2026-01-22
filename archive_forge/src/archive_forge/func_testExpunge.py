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
def testExpunge(self):
    m = SimpleMailbox()
    m.messages = [(b'Message 1', ('\\Deleted', 'AnotherFlag'), None, 0), (b'Message 2', ('AnotherFlag',), None, 1), (b'Message 3', ('\\Deleted',), None, 2)]
    SimpleServer.theAccount.addMailbox('mailbox', m)

    def login():
        return self.client.login(b'testuser', b'password-test')

    def select():
        return self.client.select('mailbox')

    def expunge():
        return self.client.expunge()

    def expunged(results):
        self.assertFalse(self.server.mbox is None)
        self.results = results
    self.results = None
    d1 = self.connected.addCallback(strip(login))
    d1.addCallbacks(strip(select), self._ebGeneral)
    d1.addCallbacks(strip(expunge), self._ebGeneral)
    d1.addCallbacks(expunged, self._ebGeneral)
    d1.addCallbacks(self._cbStopClient, self._ebGeneral)
    d2 = self.loopback()
    d = defer.gatherResults([d1, d2])
    return d.addCallback(self._cbTestExpunge, m)