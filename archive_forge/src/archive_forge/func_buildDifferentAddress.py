from __future__ import annotations
from typing import Callable
from twisted.conch.ssh.address import SSHTransportAddress
from twisted.internet.address import IPv4Address
from twisted.internet.test.test_address import AddressTestCaseMixin
from twisted.trial import unittest
def buildDifferentAddress(self) -> SSHTransportAddress:
    """
        Like C{buildAddress}, but with a different fixed address.
        """
    return SSHTransportAddress(IPv4Address('TCP', '127.0.0.2', 22))