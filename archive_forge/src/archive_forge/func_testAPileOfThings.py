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
def testAPileOfThings(self):
    SimpleServer.theAccount.addMailbox(b'inbox')
    called = []

    def login():
        called.append(None)
        return self.client.login(b'testuser', b'password-test')

    def list():
        called.append(None)
        return self.client.list(b'inbox', b'%')

    def status():
        called.append(None)
        return self.client.status(b'inbox', 'UIDNEXT')

    def examine():
        called.append(None)
        return self.client.examine(b'inbox')

    def logout():
        called.append(None)
        return self.client.logout()
    self.client.requireTransportSecurity = True
    methods = [login, list, status, examine, logout]
    for method in methods:
        self.connected.addCallback(strip(method))
    self.connected.addCallbacks(self._cbStopClient, self._ebGeneral)

    def check(ignored):
        self.assertEqual(self.server.startedTLS, True)
        self.assertEqual(self.client.startedTLS, True)
        self.assertEqual(len(called), len(methods))
    d = self.loopback()
    d.addCallback(check)
    return d