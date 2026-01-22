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
def testFetchFull(self, uid=0):
    self.function = self.client.fetchFull
    self.messages = '1,3'
    self.msgObjs = [FakeyMessage({}, ('\\XYZ', '\\YZX', 'Abc'), b'Sun, 25 Jul 2010 06:20:30 -0400 (EDT)', b'xyz' * 2, 654, None), FakeyMessage({}, ('\\One', '\\Two', 'Three'), b'Mon, 14 Apr 2003 19:43:44 -0400', b'abc' * 4, 555, None)]
    self.expected = {0: {'FLAGS': ['\\XYZ', '\\YZX', 'Abc'], 'INTERNALDATE': '25-Jul-2010 06:20:30 -0400', 'RFC822.SIZE': '6', 'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'BODY': [None, None, None, None, None, None, '6']}, 1: {'FLAGS': ['\\One', '\\Two', 'Three'], 'INTERNALDATE': '14-Apr-2003 19:43:44 -0400', 'RFC822.SIZE': '12', 'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'BODY': [None, None, None, None, None, None, '12']}}
    return self._fetchWork(uid)