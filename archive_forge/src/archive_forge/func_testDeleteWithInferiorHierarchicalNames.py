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
def testDeleteWithInferiorHierarchicalNames(self):
    """
        Attempting to delete a mailbox with hierarchically inferior
        names fails with an informative error.

        @see: U{https://tools.ietf.org/html/rfc3501#section-6.3.4}

        @return: A L{Deferred} with assertions.
        """
    SimpleServer.theAccount.addMailbox('delete')
    SimpleServer.theAccount.addMailbox('delete/me')

    def login():
        return self.client.login(b'testuser', b'password-test')

    def delete():
        return self.client.delete('delete')

    def assertIMAPException(failure):
        failure.trap(imap4.IMAP4Exception)
        self.assertEqual(str(failure.value), str(b'Name "DELETE" has inferior hierarchical names'))
    loggedIn = self.connected.addCallback(strip(login))
    loggedIn.addCallbacks(strip(delete), self._ebGeneral)
    loggedIn.addErrback(assertIMAPException)
    loggedIn.addCallbacks(self._cbStopClient)
    loopedBack = self.loopback()
    d = defer.gatherResults([loggedIn, loopedBack])
    d.addCallback(lambda _: self.assertEqual(sorted(SimpleServer.theAccount.mailboxes), ['DELETE', 'DELETE/ME']))
    return d