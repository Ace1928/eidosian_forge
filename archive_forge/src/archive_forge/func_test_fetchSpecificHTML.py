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
def test_fetchSpecificHTML(self):
    """
        If the body of a message begins with I{<} and ends with I{>} (as,
        for example, HTML bodies typically will), this is still interpreted
        as the body by L{IMAP4Client.fetchSpecific} (and particularly, not
        as a length indicator for a response to a request for a partial
        body).
        """
    d = self.client.fetchSpecific('7')
    self.assertEqual(self.transport.value(), b'0001 FETCH 7 BODY[]\r\n')
    self.client.lineReceived(b'* 7 FETCH (BODY[] "<html>test</html>")')
    self.client.lineReceived(b'0001 OK FETCH completed')
    self.assertEqual(self.successResultOf(d), {7: [['BODY', [], '<html>test</html>']]})