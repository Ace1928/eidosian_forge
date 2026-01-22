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
def test_bufferedServerStatus(self):
    """
        When a server status change occurs during an ongoing FETCH
        command, the server status is buffered until the FETCH
        completes.
        """
    self.server.dataReceived(b'01 FETCH 1,2 BODY[]\r\n')
    twice = functools.partial(next, iter([True, True, False]))
    self.flushPending(asLongAs=twice)
    self.assertEqual(self.transport.value(), b''.join([b'* 1 FETCH (BODY[] )\r\n', networkString('{5}\r\n\r\n\r\n%s' % (nativeString(self.messages[0].getBodyFile().read()),))]))
    self.transport.clear()
    self.server.modeChanged(writeable=True)
    self.assertFalse(self.transport.value())
    self.flushPending()
    self.assertEqual(self.transport.value(), b''.join([b'* 2 FETCH (BODY[] )\r\n', b'* [READ-WRITE]\r\n', networkString('01 OK FETCH completed\r\n{5}\r\n\r\n\r\n%s' % (nativeString(self.messages[1].getBodyFile().read()),))]))