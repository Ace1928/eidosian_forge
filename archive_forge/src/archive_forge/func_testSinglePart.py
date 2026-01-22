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
def testSinglePart(self):
    body = b'This is body text.  Rar.'
    headers = OrderedDict()
    headers['from'] = 'sender@host'
    headers['to'] = 'recipient@domain'
    headers['subject'] = 'booga booga boo'
    headers['content-type'] = 'text/plain'
    msg = FakeyMessage(headers, (), None, body, 123, None)
    c = BufferingConsumer()
    p = imap4.MessageProducer(msg)
    d = p.beginProducing(c)

    def cbProduced(result):
        self.assertIdentical(result, p)
        self.assertEqual(b''.join(c.buffer), b'{119}\r\nFrom: sender@host\r\nTo: recipient@domain\r\nSubject: booga booga boo\r\nContent-Type: text/plain\r\n\r\n' + body)
    return d.addCallback(cbProduced)