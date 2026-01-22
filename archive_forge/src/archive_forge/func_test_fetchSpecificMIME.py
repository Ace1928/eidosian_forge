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
def test_fetchSpecificMIME(self):
    """
        L{IMAP4Client.fetchSpecific}, when passed C{'MIME'} for C{headerType},
        sends the I{BODY[MIME]} command.  It returns a L{Deferred} which fires
        with a C{dict} mapping message sequence numbers to C{list}s of
        corresponding message data given by the server's response.
        """
    d = self.client.fetchSpecific('8', headerType='MIME')
    self.assertEqual(self.transport.value(), b'0001 FETCH 8 BODY[MIME]\r\n')
    self.client.lineReceived(b'* 8 FETCH (BODY[MIME] "Some body")')
    self.client.lineReceived(b'0001 OK FETCH completed')
    self.assertEqual(self.successResultOf(d), {8: [['BODY', ['MIME'], 'Some body']]})