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
def getHandleErrorCodeMatcher(self):
    """
        Return a L{hamcrest.core.matcher.Matcher} that matches the
        errno expected to result from writing to a closed platform
        socket handle.
        """
    if platform.isWindows():
        return hamcrest.equal_to(errno.WSAENOTSOCK)
    return hamcrest.equal_to(errno.EBADF)