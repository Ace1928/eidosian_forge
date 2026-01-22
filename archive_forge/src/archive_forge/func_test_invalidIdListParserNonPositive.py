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
def test_invalidIdListParserNonPositive(self):
    """
        Zeroes and negative values are not accepted in id range expressions. RFC
        3501 states that sequence numbers and sequence ranges consist of
        non-negative numbers (RFC 3501 section 9, the seq-number grammar item).
        """
    inputs = [b'0:5', b'0:0', b'*:0', b'0', b'-3:5', b'1:-2', b'-1']
    for input in inputs:
        self.assertRaises(imap4.IllegalIdentifierError, imap4.parseIdList, input, 12345)