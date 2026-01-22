from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def test_connectionLostSubclassOfConnectionClosed(self) -> None:
    """
        L{error.ConnectionClosed} is a superclass of L{error.ConnectionLost}.
        """
    self.assertTrue(issubclass(error.ConnectionLost, error.ConnectionClosed))