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
def test_defaultPeerAddress(self) -> None:
    """
        If no peer address is passed to L{StringTransport.__init__}, an
        L{IPv4Address} is returned from L{StringTransport.getPeer}.
        """
    address = StringTransport().getPeer()
    self.assertIsInstance(address, IPv4Address)