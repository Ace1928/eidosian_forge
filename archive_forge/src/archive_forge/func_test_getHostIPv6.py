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
@skipWithoutIPv6
def test_getHostIPv6(self):
    """
        L{IListeningPort.getHost} returns an L{IPv6Address} when listening on
        an IPv6 interface.
        """
    reactor = self.buildReactor()
    port = self.getListeningPort(reactor, DatagramProtocol(), interface='::1')
    addr = port.getHost()
    self.assertEqual(addr.host, '::1')
    self.assertIsInstance(addr, IPv6Address)