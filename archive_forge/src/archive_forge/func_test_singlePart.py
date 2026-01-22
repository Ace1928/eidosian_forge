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
def test_singlePart(self):
    """
        L{imap4.getBodyStructure} accepts a L{IMessagePart} provider and returns
        a list giving the basic fields for the I{BODY} response for that
        message.
        """
    body = b'hello, world'
    major = 'image'
    minor = 'jpeg'
    charset = 'us-ascii'
    identifier = 'some kind of id'
    description = 'great justice'
    encoding = 'maximum'
    msg = FakeyMessage({'content-type': major + '/' + minor + '; charset=' + charset + '; x=y', 'content-id': identifier, 'content-description': description, 'content-transfer-encoding': encoding}, (), b'', body, 123, None)
    structure = imap4.getBodyStructure(msg)
    self.assertEqual([major, minor, ['charset', charset, 'x', 'y'], identifier, description, encoding, len(body)], structure)