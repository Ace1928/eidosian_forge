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
def test_idleClientDoesDisconnect(self):
    """
        The *server* has a timeout mechanism which will close connections that
        are inactive for a period.
        """
    c = Clock()
    transport = StringTransportWithDisconnection()
    transport.protocol = self.server
    self.server.callLater = c.callLater
    self.server.makeConnection(transport)
    lost = []
    connLost = self.server.connectionLost
    self.server.connectionLost = lambda reason: (lost.append(None), connLost(reason))[1]
    c.pump([0.0] + [self.server.timeOut / 3.0] * 2)
    self.assertFalse(lost, lost)
    c.pump([0.0, self.server.timeOut / 2.0])
    self.assertTrue(lost)