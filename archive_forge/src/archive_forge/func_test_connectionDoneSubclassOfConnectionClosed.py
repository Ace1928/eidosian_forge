from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def test_connectionDoneSubclassOfConnectionClosed(self) -> None:
    """
        L{error.ConnectionClosed} is a superclass of L{error.ConnectionDone}.
        """
    self.assertTrue(issubclass(error.ConnectionDone, error.ConnectionClosed))