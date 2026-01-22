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
def testPartialRetrieve(self):
    p, t = setUp()
    d = p.retrieve(7, lines=2)
    self.assertEqual(t.value(), b'TOP 8 2\r\n')
    p.dataReceived(b'+OK 2 lines on the way\r\n')
    p.dataReceived(b'Line the first!  Woop\r\n')
    p.dataReceived(b'Line the last!  Bye\r\n')
    p.dataReceived(b'.\r\n')
    return d.addCallback(self.assertEqual, [b'Line the first!  Woop', b'Line the last!  Bye'])