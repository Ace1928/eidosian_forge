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
def test_fetchParserSimple(self):
    cases = [['ENVELOPE', 'Envelope', 'envelope'], ['FLAGS', 'Flags', 'flags'], ['INTERNALDATE', 'InternalDate', 'internaldate'], ['RFC822.HEADER', 'RFC822Header', 'rfc822.header'], ['RFC822.SIZE', 'RFC822Size', 'rfc822.size'], ['RFC822.TEXT', 'RFC822Text', 'rfc822.text'], ['RFC822', 'RFC822', 'rfc822'], ['UID', 'UID', 'uid'], ['BODYSTRUCTURE', 'BodyStructure', 'bodystructure']]
    for inp, outp, asString in cases:
        inp = inp.encode('ascii')
        p = imap4._FetchParser()
        p.parseString(inp)
        self.assertEqual(len(p.result), 1)
        self.assertTrue(isinstance(p.result[0], getattr(p, outp)))
        self.assertEqual(str(p.result[0]), asString)