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
def test_incompleteFetchSpecificTextResponse(self):
    """
        If the server responds to a I{BODY[TEXT]} request with a I{FETCH} line
        which is truncated after the I{BODY[TEXT]} tokens, the L{Deferred}
        returned by L{IMAP4Client.fetchUID} fails with
        L{IllegalServerResponse}.
        """
    d = self.client.fetchSpecific('8', headerType='TEXT')
    self.assertEqual(self.transport.value(), b'0001 FETCH 8 BODY[TEXT]\r\n')
    self.client.lineReceived(b'* 8 FETCH (BODY[TEXT])')
    self.client.lineReceived(b'0001 OK FETCH completed')
    self.failureResultOf(d, imap4.IllegalServerResponse)