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
def testFailedCramMD5(self):
    self.server.challengers[b'CRAM-MD5'] = CramMD5Credentials
    cAuth = imap4.CramMD5ClientAuthenticator(b'testuser')
    self.client.registerAuthenticator(cAuth)

    def misauth():
        return self.client.authenticate(b'not the secret')

    def authed():
        self.authenticated = 1

    def misauthed():
        self.authenticated = -1
    d1 = self.connected.addCallback(strip(misauth))
    d1.addCallbacks(strip(authed), strip(misauthed))
    d1.addCallbacks(self._cbStopClient, self._ebGeneral)
    d = defer.gatherResults([self.loopback(), d1])
    return d.addCallback(self._cbTestFailedCramMD5)