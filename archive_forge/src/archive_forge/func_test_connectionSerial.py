from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def test_connectionSerial(self) -> None:
    """
        Each L{FakeTransport} receives a serial number that uniquely identifies
        it.
        """
    a = FakeTransport(object(), True)
    b = FakeTransport(object(), False)
    self.assertIsInstance(a.serial, int)
    self.assertIsInstance(b.serial, int)
    self.assertNotEqual(a.serial, b.serial)