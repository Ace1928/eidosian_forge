import socket
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import defer, error
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import DatagramProtocol
from twisted.internet.test.connectionmixins import LogObserverMixin, findFreePort
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python import context
from twisted.python.log import ILogContext, err
from twisted.test.test_udp import GoodClient, Server
from twisted.trial.unittest import SkipTest
def test_invalidInterface(self):
    """
        An L{InvalidAddressError} is raised when trying to listen on an address
        that isn't a valid IPv4 or IPv6 address.
        """
    reactor = self.buildReactor()
    self.assertRaises(error.InvalidAddressError, reactor.listenUDP, DatagramProtocol(), 0, interface='example.com')