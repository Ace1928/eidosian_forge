from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def test_errno(self) -> None:
    """
        L{error.getConnectError} converts based on errno for C{socket.error}.
        """
    self.assertErrnoException(errno.ENETUNREACH, error.NoRouteError)
    self.assertErrnoException(errno.ECONNREFUSED, error.ConnectionRefusedError)
    self.assertErrnoException(errno.ETIMEDOUT, error.TCPTimedOutError)
    if sys.platform == 'win32':
        self.assertErrnoException(errno.WSAECONNREFUSED, error.ConnectionRefusedError)
        self.assertErrnoException(errno.WSAENETUNREACH, error.NoRouteError)