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
def testShutdownException(self):
    """
        If the other side has already closed its connection,
        loseWriteConnection should pass silently.
        """
    self.f.protocol.transport.loseConnection()
    self.client.transport.write(b'X')
    self.client.transport.loseWriteConnection()
    self.f.protocol.closedDeferred = d = defer.Deferred()
    self.client.closedDeferred = d2 = defer.Deferred()
    d.addCallback(lambda x: self.assertTrue(self.f.protocol.closed))
    return defer.gatherResults([d, d2])