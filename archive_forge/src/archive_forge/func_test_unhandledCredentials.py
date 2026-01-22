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
def test_unhandledCredentials(self):
    """
        A challenger that causes the login to fail
        L{UnhandledCredentials} results in an C{NO} response.

        @return: A L{Deferred} that fires when the authorization has
            failed.
        """
    realm = TestRealm()
    portal = Portal(realm)
    self.server.portal = portal
    self.server.challengers[b'LOGIN'] = loginCred = imap4.LOGINCredentials
    verifyClass(IChallengeResponse, loginCred)
    cAuth = imap4.LOGINAuthenticator(b'testuser')
    self.client.registerAuthenticator(cAuth)

    def auth():
        return self.client.authenticate(b'secret')
    d1 = self.connected.addCallback(strip(auth))
    d1.addErrback(self.assertClientFailureMessage, b'Authentication failed: server misconfigured')
    d1.addCallbacks(self._cbStopClient, self._ebGeneral)
    d = defer.gatherResults([self.loopback(), d1])
    return d