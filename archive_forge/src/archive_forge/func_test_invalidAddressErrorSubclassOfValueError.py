from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def test_invalidAddressErrorSubclassOfValueError(self) -> None:
    """
        L{ValueError} is a superclass of L{error.InvalidAddressError}.
        """
    self.assertTrue(issubclass(error.InvalidAddressError, ValueError))