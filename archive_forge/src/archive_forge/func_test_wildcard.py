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
def test_wildcard(self):
    cases = [['foo/%gum/bar', ['foo/bar', 'oo/lalagum/bar', 'foo/gumx/bar', 'foo/gum/baz'], ['foo/xgum/bar', 'foo/gum/bar']], ['foo/x%x/bar', ['foo', 'bar', 'fuz fuz fuz', 'foo/*/bar', 'foo/xyz/bar', 'foo/xx/baz'], ['foo/xyx/bar', 'foo/xx/bar', 'foo/xxxxxxxxxxxxxx/bar']], ['foo/xyz*abc/bar', ['foo/xyz/bar', 'foo/abc/bar', 'foo/xyzab/cbar', 'foo/xyza/bcbar'], ['foo/xyzabc/bar', 'foo/xyz/abc/bar', 'foo/xyz/123/abc/bar']]]
    for wildcard, fail, succeed in cases:
        wildcard = imap4.wildcardToRegexp(wildcard, '/')
        for x in fail:
            self.assertFalse(wildcard.match(x))
        for x in succeed:
            self.assertTrue(wildcard.match(x))