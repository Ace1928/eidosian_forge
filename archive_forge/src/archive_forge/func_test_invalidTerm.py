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
def test_invalidTerm(self):
    """
        If, as part of a search, an ISearchableMailbox raises an
        IllegalQueryError (e.g. due to invalid search criteria), client sees a
        failure response, and an IllegalQueryError is logged on the server.
        """
    query = 'FOO'

    def search():
        return self.client.search(query)
    d = self.connected.addCallback(strip(search))
    d = self.assertFailure(d, imap4.IMAP4Exception)

    def errorReceived(results):
        """
            Verify that the server logs an IllegalQueryError and the
            client raises an IMAP4Exception with 'Search failed:...'
            """
        self.client.transport.loseConnection()
        self.server.transport.loseConnection()
        errors = self.flushLoggedErrors(imap4.IllegalQueryError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(str(b'SEARCH failed: FOO is not a valid search criteria'), str(results))
    d.addCallback(errorReceived)
    d.addErrback(self._ebGeneral)
    self.loopback()
    return d