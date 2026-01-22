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
def testCapabilityWithAuth(self):
    caps = {}
    self.server.challengers[b'CRAM-MD5'] = CramMD5Credentials

    def getCaps():

        def gotCaps(c):
            caps.update(c)
            self.server.transport.loseConnection()
        return self.client.getCapabilities().addCallback(gotCaps)
    d1 = self.connected.addCallback(strip(getCaps)).addErrback(self._ebGeneral)
    d = defer.gatherResults([self.loopback(), d1])
    expCap = {b'IMAP4rev1': None, b'NAMESPACE': None, b'IDLE': None, b'AUTH': [b'CRAM-MD5']}
    return d.addCallback(lambda _: self.assertEqual(expCap, caps))