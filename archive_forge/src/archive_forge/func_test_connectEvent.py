import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
def test_connectEvent(self):
    """
        This test checks that we correctly get notifications event for a
        client.  This ought to prevent a regression under Windows using the
        GTK2 reactor.  See #3925.
        """
    reactor = self.buildReactor()
    self.listen(reactor, ServerFactory.forProtocol(Protocol))
    connected = []

    class CheckConnection(Protocol):

        def connectionMade(self):
            connected.append(self)
            reactor.stop()
    clientFactory = Stop(reactor)
    clientFactory.protocol = CheckConnection
    needsRunningReactor(reactor, lambda: self.connect(reactor, clientFactory))
    reactor.run()
    self.assertTrue(connected)