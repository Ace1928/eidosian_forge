import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbSendsFinished(ignored):
    cAddr = client.transport.getHost()
    sAddr = server.transport.getHost()
    self.assertEqual(client.packets, [(b'hello', (sAddr.host, sAddr.port))])
    clientAddr = (cAddr.host, cAddr.port)
    self.assertEqual(server.packets, [(b'a', clientAddr), (b'b', clientAddr), (b'c', clientAddr)])