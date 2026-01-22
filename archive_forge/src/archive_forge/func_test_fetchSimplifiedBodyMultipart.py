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
def test_fetchSimplifiedBodyMultipart(self):
    """
        L{IMAP4Client.fetchSimplifiedBody} returns a dictionary mapping message
        sequence numbers to fetch responses for the corresponding messages.  In
        particular, for a multipart message, the value in the dictionary maps
        the string C{"BODY"} to a list giving the body structure information for
        that message, in the form of a list of subpart body structure
        information followed by the subtype of the message (eg C{"alternative"}
        for a I{multipart/alternative} message).  This structure is self-similar
        in the case where a subpart is itself multipart.
        """
    self.function = self.client.fetchSimplifiedBody
    self.messages = '21'
    singles = [FakeyMessage({'content-type': 'text/plain'}, (), b'date', b'Stuff', 54321, None), FakeyMessage({'content-type': 'text/html'}, (), b'date', b'Things', 32415, None)]
    alternative = FakeyMessage({'content-type': 'multipart/alternative'}, (), b'', b'Irrelevant', 12345, singles)
    mixed = FakeyMessage({'content-type': 'multipart/mixed'}, (), b'', b'RootOf', 98765, [alternative])
    self.msgObjs = [mixed]
    self.expected = {0: {'BODY': [[['text', 'plain', None, None, None, None, '5', '1'], ['text', 'html', None, None, None, None, '6', '1'], 'alternative'], 'mixed']}}
    return self._fetchWork(False)