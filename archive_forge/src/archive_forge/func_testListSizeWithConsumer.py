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
def testListSizeWithConsumer(self):
    p, t = setUp()
    c = ListConsumer()
    f = c.consume
    d = p.listSize(f)
    self.assertEqual(t.value(), b'LIST\r\n')
    p.dataReceived(b'+OK Here it comes\r\n')
    p.dataReceived(b'1 3\r\n2 2\r\n3 1\r\n')
    self.assertEqual(c.data, {0: [3], 1: [2], 2: [1]})
    p.dataReceived(b'5 3\r\n6 2\r\n7 1\r\n')
    self.assertEqual(c.data, {0: [3], 1: [2], 2: [1], 4: [3], 5: [2], 6: [1]})
    p.dataReceived(b'.\r\n')
    return d.addCallback(self.assertIdentical, f)