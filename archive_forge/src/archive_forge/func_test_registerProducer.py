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
def test_registerProducer(self) -> None:
    """
        L{StringTransport.registerProducer} records the arguments supplied to
        it as instance attributes.
        """
    producer = object()
    streaming = object()
    self.transport.registerProducer(producer, streaming)
    self.assertIs(self.transport.producer, producer)
    self.assertIs(self.transport.streaming, streaming)