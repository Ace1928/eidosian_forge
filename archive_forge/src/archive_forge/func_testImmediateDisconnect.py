import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def testImmediateDisconnect(self):
    org = 'twisted.test.test_ssl'
    self.setupServerAndClient((org, org + ', client'), {}, (org, org + ', server'), {})
    serverProtocolFactory = protocol.ServerFactory()
    serverProtocolFactory.protocol = protocol.Protocol
    self.serverPort = serverPort = reactor.listenSSL(0, serverProtocolFactory, self.serverCtxFactory)
    clientProtocolFactory = protocol.ClientFactory()
    clientProtocolFactory.protocol = ImmediatelyDisconnectingProtocol
    clientProtocolFactory.connectionDisconnected = defer.Deferred()
    reactor.connectSSL('127.0.0.1', serverPort.getHost().port, clientProtocolFactory, self.clientCtxFactory)
    return clientProtocolFactory.connectionDisconnected.addCallback(lambda ignoredResult: self.serverPort.stopListening())