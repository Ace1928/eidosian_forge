from __future__ import annotations
from typing import Callable
from twisted.conch.ssh.address import SSHTransportAddress
from twisted.internet.address import IPv4Address
from twisted.internet.test.test_address import AddressTestCaseMixin
from twisted.trial import unittest
def buildAddress(self) -> SSHTransportAddress:
    """
        Create an arbitrary new C{SSHTransportAddress}.  A new instance is
        created for each call, but always for the same address.
        """
    return SSHTransportAddress(IPv4Address('TCP', '127.0.0.1', 22))