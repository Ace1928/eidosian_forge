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
def test_startTLSWithExistingChallengers(self):
    """
        Starting a TLS negotiation with an L{IMAP4Server} that already
        has C{LOGIN} and C{PLAIN} L{IChallengeResponse} factories uses
        those factories.
        """
    self.server.challengers = {b'LOGIN': imap4.LOGINCredentials, b'PLAIN': imap4.PLAINCredentials}

    @defer.inlineCallbacks
    def assertLOGINandPLAIN():
        capabilities = (yield self.client.getCapabilities())
        self.assertIn(b'AUTH', capabilities)
        self.assertIn(b'LOGIN', capabilities[b'AUTH'])
        self.assertIn(b'PLAIN', capabilities[b'AUTH'])
    self.connected.addCallback(strip(assertLOGINandPLAIN))
    disconnected = self.startTLSAndAssertSession()
    self.connected.addCallback(strip(assertLOGINandPLAIN))
    self.connected.addCallback(self._cbStopClient)
    self.connected.addErrback(self._ebGeneral)
    return disconnected