from __future__ import annotations
from typing import Callable
from zope.interface.verify import verifyObject
from typing_extensions import Protocol
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory
from twisted.internet.testing import (
from twisted.python.reflect import namedAny
from twisted.trial.unittest import TestCase
def test_disconnectingTransportCannotResume(self) -> None:
    """
        L{StringTransport.resumeProducing} raises L{RuntimeError} if the
        transport is being disconnected.
        """
    self.transport.loseConnection()
    self.assertRaises(RuntimeError, self.transport.resumeProducing)