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
def testPrivileged(self):
    factory = protocol.ServerFactory()
    factory.protocol = TestEcho
    TestEcho.d = defer.Deferred()
    t = internet.TCPServer(0, factory)
    t.privileged = 1
    t.privilegedStartService()
    num = t._port.getHost().port
    factory = protocol.ClientFactory()
    factory.d = defer.Deferred()
    factory.protocol = Foo
    factory.line = None
    c = internet.TCPClient('127.0.0.1', num, factory)
    c.startService()
    factory.d.addCallback(self.assertEqual, b'lalala')
    factory.d.addCallback(lambda x: c.stopService())
    factory.d.addCallback(lambda x: t.stopService())
    factory.d.addCallback(lambda x: TestEcho.d)
    return factory.d