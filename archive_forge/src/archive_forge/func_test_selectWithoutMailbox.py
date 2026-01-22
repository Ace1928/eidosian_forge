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
def test_selectWithoutMailbox(self):
    """
        A client that selects a mailbox that does not exist receives a
        C{NO} response.
        """

    def login():
        return self.client.login(b'testuser', b'password-test')

    def select():
        return self.client.select('test-mailbox')
    self.connected.addCallback(strip(login))
    self.connected.addCallback(strip(select))
    self.connected.addErrback(self.assertClientFailureMessage, b'No such mailbox')
    self.connected.addCallback(self._cbStopClient)
    self.connected.addErrback(self._ebGeneral)
    connectionComplete = defer.gatherResults([self.connected, self.loopback()])

    @connectionComplete.addCallback
    def assertNoMailboxSelected(_):
        self.assertIsNone(self.server.mbox)
    return connectionComplete