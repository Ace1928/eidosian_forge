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
def test_concurrentRetrieves(self):
    """
        Issue three retrieve calls immediately without waiting for any to
        succeed and make sure they all do succeed eventually.
        """
    p, t = setUp()
    messages = [p.retrieve(i).addCallback(self.assertEqual, [b'First line of %d.' % (i + 1,), b'Second line of %d.' % (i + 1,)]) for i in range(3)]
    for i in range(1, 4):
        self.assertEqual(t.value(), b'RETR %d\r\n' % (i,))
        t.clear()
        p.dataReceived(b'+OK 2 lines on the way\r\n')
        p.dataReceived(b'First line of %d.\r\n' % (i,))
        p.dataReceived(b'Second line of %d.\r\n' % (i,))
        self.assertEqual(t.value(), b'')
        p.dataReceived(b'.\r\n')
    return defer.DeferredList(messages, fireOnOneErrback=True)