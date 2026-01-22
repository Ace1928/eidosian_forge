import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_reactorParametrizationInClientMultipleStart(self):
    """
        Like L{test_reactorParametrizationInClient}, but stop and restart the
        service and check that the given reactor is still used.
        """
    reactor = MemoryReactor()
    factory = protocol.ClientFactory()
    t = internet.TCPClient('127.0.0.1', 1234, factory, reactor=reactor)
    t.startService()
    self.assertEqual(reactor.tcpClients.pop()[:3], ('127.0.0.1', 1234, factory))
    t.stopService()
    t.startService()
    self.assertEqual(reactor.tcpClients.pop()[:3], ('127.0.0.1', 1234, factory))