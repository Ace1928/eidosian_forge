import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
def testInsecureLoginRaisesException(self):
    p, t = setUp(greet=False)
    p.dataReceived(b'+OK Howdy\r\n')
    d = p.login(b'username', b'password')
    self.assertFalse(t.value())
    return self.assertFailure(d, InsecureAuthenticationDisallowed)