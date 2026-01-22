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
def test_writeNotificationRaises(self):
    """
        If C{writeConnectionLost} raises an exception when the transport
        calls it to notify the protocol of that event, the exception should
        be logged and the protocol should be disconnected completely.
        """
    self.clientProtocol.writeConnectionLost = self.aBug
    return self._notificationRaisesTest()