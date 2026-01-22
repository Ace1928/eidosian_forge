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
def testFetchAll(self, uid=0):
    self.function = self.client.fetchAll
    self.messages = '1,2:3'
    self.msgObjs = [FakeyMessage({}, (), b'Mon, 14 Apr 2003 19:43:44 +0400', b'Lalala', 10101, None), FakeyMessage({}, (), b'Tue, 15 Apr 2003 19:43:44 +0200', b'Alalal', 20202, None)]
    self.expected = {0: {'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'RFC822.SIZE': '6', 'INTERNALDATE': '14-Apr-2003 19:43:44 +0400', 'FLAGS': []}, 1: {'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'RFC822.SIZE': '6', 'INTERNALDATE': '15-Apr-2003 19:43:44 +0200', 'FLAGS': []}}
    return self._fetchWork(uid)