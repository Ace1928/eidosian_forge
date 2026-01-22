import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def test_serverRepr(self):
    """
        Check that the repr string of the server transport get the good port
        number if the server listens on 0.
        """
    server = MyServerFactory()
    serverConnMade = server.protocolConnectionMade = defer.Deferred()
    port = reactor.listenTCP(0, server)
    self.addCleanup(port.stopListening)
    client = MyClientFactory()
    clientConnMade = client.protocolConnectionMade = defer.Deferred()
    connector = reactor.connectTCP('127.0.0.1', port.getHost().port, client)
    self.addCleanup(connector.disconnect)

    def check(result):
        serverProto, clientProto = result
        portNumber = port.getHost().port
        self.assertEqual(repr(serverProto.transport), f'<AccumulatingProtocol #0 on {portNumber}>')
        serverProto.transport.loseConnection()
        clientProto.transport.loseConnection()
    return defer.gatherResults([serverConnMade, clientConnMade]).addCallback(check)