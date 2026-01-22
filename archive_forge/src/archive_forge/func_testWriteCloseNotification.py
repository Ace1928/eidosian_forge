import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def testWriteCloseNotification(self):
    f = self.f
    f.protocol.transport.loseWriteConnection()
    d = defer.gatherResults([loopUntil(lambda: f.protocol.writeHalfClosed), loopUntil(lambda: self.client.readHalfClosed)])
    d.addCallback(lambda _: self.assertEqual(f.protocol.readHalfClosed, False))
    return d