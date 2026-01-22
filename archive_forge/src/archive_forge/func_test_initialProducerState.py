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
def test_initialProducerState(self) -> None:
    """
        L{StringTransport.producerState} is initially C{'producing'}.
        """
    self.assertEqual(self.transport.producerState, 'producing')