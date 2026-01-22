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
def testSetFlags(self, uid=0):
    self.function = self.client.setFlags
    self.messages = '1,5,9'
    self.flags = ['\\A', '\\B', 'C']
    self.silent = False
    self.uid = uid
    self.response = {1: ['\\A', '\\B', 'C'], 5: ['\\A', '\\B', 'C'], 9: ['\\A', '\\B', 'C']}
    self.expected = {1: {'FLAGS': ['\\A', '\\B', 'C']}, 5: {'FLAGS': ['\\A', '\\B', 'C']}, 9: {'FLAGS': ['\\A', '\\B', 'C']}}
    msg = imap4.MessageSet()
    msg.add(1)
    msg.add(5)
    msg.add(9)
    self.expectedArgs = ((msg, ['\\A', '\\B', 'C'], 0), {'uid': 0})
    return self._storeWork()