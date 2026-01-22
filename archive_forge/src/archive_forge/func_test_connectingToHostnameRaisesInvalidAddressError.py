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
def test_connectingToHostnameRaisesInvalidAddressError(self):
    """
        Connecting to a hostname instead of an IP address will raise an
        L{InvalidAddressError}.
        """
    reactor = self.buildReactor()
    port = self.getListeningPort(reactor, DatagramProtocol())
    self.assertRaises(error.InvalidAddressError, port.connect, 'example.invalid', 1)