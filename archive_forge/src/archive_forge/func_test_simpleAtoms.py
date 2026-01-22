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
def test_simpleAtoms(self):
    """
        A capability response consisting only of atoms without C{'='} in them
        should result in a dict mapping those atoms to L{None}.
        """
    capabilitiesResult = self.protocol.getCapabilities(useCache=False)
    self.protocol.dataReceived(b'* CAPABILITY IMAP4rev1 LOGINDISABLED\r\n')
    self.protocol.dataReceived(b'0001 OK Capability completed.\r\n')

    def gotCapabilities(capabilities):
        self.assertEqual(capabilities, {b'IMAP4rev1': None, b'LOGINDISABLED': None})
    capabilitiesResult.addCallback(gotCapabilities)
    return capabilitiesResult