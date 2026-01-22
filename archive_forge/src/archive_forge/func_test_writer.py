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
def test_writer(self):
    """
        L{ITCPTransport.write} and L{ITCPTransport.writeSequence} send bytes to
        the other end of the connection.
        """
    f = protocol.Factory()
    f.protocol = WriterProtocol
    f.done = 0
    f.problem = 0
    wrappedF = WiredFactory(f)
    p = reactor.listenTCP(0, wrappedF, interface='127.0.0.1')
    self.addCleanup(p.stopListening)
    n = p.getHost().port
    clientF = WriterClientFactory()
    wrappedClientF = WiredFactory(clientF)
    reactor.connectTCP('127.0.0.1', n, wrappedClientF)

    def check(ignored):
        self.assertTrue(f.done, "writer didn't finish, it probably died")
        self.assertTrue(f.problem == 0, 'writer indicated an error')
        self.assertTrue(clientF.done, "client didn't see connection dropped")
        expected = b''.join([b'Hello Cleveland!\n', b'Goodbye', b' cruel', b' world', b'\n'])
        self.assertTrue(clientF.data == expected, "client didn't receive all the data it expected")
    d = defer.gatherResults([wrappedF.onDisconnect, wrappedClientF.onDisconnect])
    return d.addCallback(check)