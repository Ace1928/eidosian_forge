import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbStarted(ignored):
    connectionRefused = client.startedDeferred = defer.Deferred()
    client.transport.connect('127.0.0.1', 80)
    for i in range(10):
        client.transport.write(b'%d' % (i,))
        server.transport.write(b'%d' % (i,), ('127.0.0.1', 80))
    return self.assertFailure(connectionRefused, error.ConnectionRefusedError)