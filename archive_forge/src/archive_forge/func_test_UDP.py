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
def test_UDP(self):
    """
        Test L{internet.UDPServer} with a random port: starting the service
        should give it valid port, and stopService should free it so that we
        can start a server on the same port again.
        """
    if not interfaces.IReactorUDP(reactor, None):
        raise SkipTest('This reactor does not support UDP sockets')
    p = protocol.DatagramProtocol()
    t = internet.UDPServer(0, p)
    t.startService()
    num = t._port.getHost().port
    self.assertNotEqual(num, 0)

    def onStop(ignored):
        t = internet.UDPServer(num, p)
        t.startService()
        return t.stopService()
    return defer.maybeDeferred(t.stopService).addCallback(onStop)