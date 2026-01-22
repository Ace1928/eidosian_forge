import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbInterfaces(ignored):
    self.assertEqual(self.client.transport.getOutgoingInterface(), '127.0.0.1')
    self.assertEqual(self.server.transport.getOutgoingInterface(), '127.0.0.1')