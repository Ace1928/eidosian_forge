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
def clientDisconnected(result):
    """
            Verify that the underlying platform socket handle has been
            cleaned up.
            """
    client, server = result
    if not client.lostConnectionReason.check(error.ConnectionClosed):
        err(client.lostConnectionReason, 'Client lost connection for unexpected reason')
    if not server.lostConnectionReason.check(error.ConnectionClosed):
        err(server.lostConnectionReason, 'Server lost connection for unexpected reason')
    errorCodeMatcher = self.getHandleErrorCodeMatcher()
    exception = self.assertRaises(self.getHandleExceptionType(), client.handle.send, b'bytes')
    hamcrest.assert_that(exception.args[0], errorCodeMatcher)