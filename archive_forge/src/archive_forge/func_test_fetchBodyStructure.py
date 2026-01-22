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
def test_fetchBodyStructure(self, uid=0):
    """
        L{IMAP4Client.fetchBodyStructure} issues a I{FETCH BODYSTRUCTURE}
        command and returns a Deferred which fires with a structure giving the
        result of parsing the server's response.  The structure is a list
        reflecting the parenthesized data sent by the server, as described by
        RFC 3501, section 7.4.2.
        """
    self.function = self.client.fetchBodyStructure
    self.messages = '3:9,10:*'
    self.msgObjs = [FakeyMessage({'content-type': 'text/plain; name=thing; key="value"', 'content-id': 'this-is-the-content-id', 'content-description': 'describing-the-content-goes-here!', 'content-transfer-encoding': '8BIT', 'content-md5': 'abcdef123456', 'content-disposition': 'attachment; filename=monkeys', 'content-language': 'es', 'content-location': 'http://example.com/monkeys'}, (), '', b'Body\nText\nGoes\nHere\n', 919293, None)]
    self.expected = {0: {'BODYSTRUCTURE': ['text', 'plain', ['key', 'value', 'name', 'thing'], 'this-is-the-content-id', 'describing-the-content-goes-here!', '8BIT', '20', '4', 'abcdef123456', ['attachment', ['filename', 'monkeys']], 'es', 'http://example.com/monkeys']}}
    return self._fetchWork(uid)