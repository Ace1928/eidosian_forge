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
def runProtocolsWithReactor(reactorBuilder, serverProtocol, clientProtocol, endpointCreator):
    """
    Connect two protocols using endpoints and a new reactor instance.

    A new reactor will be created and run, with the client and server protocol
    instances connected to each other using the given endpoint creator. The
    protocols should run through some set of tests, then disconnect; when both
    have disconnected the reactor will be stopped and the function will
    return.

    @param reactorBuilder: A L{ReactorBuilder} instance.

    @param serverProtocol: A L{ConnectableProtocol} that will be the server.

    @param clientProtocol: A L{ConnectableProtocol} that will be the client.

    @param endpointCreator: An instance of L{EndpointCreator}.

    @return: The reactor run by this test.
    """
    reactor = reactorBuilder.buildReactor()
    serverProtocol._setAttributes(reactor, Deferred())
    clientProtocol._setAttributes(reactor, Deferred())
    serverFactory = _SingleProtocolFactory(serverProtocol)
    clientFactory = _SingleProtocolFactory(clientProtocol)
    serverEndpoint = endpointCreator.server(reactor)
    d = serverEndpoint.listen(serverFactory)

    def gotPort(p):
        clientEndpoint = endpointCreator.client(reactor, p.getHost())
        return clientEndpoint.connect(clientFactory)
    d.addCallback(gotPort)

    def failed(result):
        log.err(result, 'Connection setup failed.')
    disconnected = gatherResults([serverProtocol._done, clientProtocol._done])
    d.addCallback(lambda _: disconnected)
    d.addErrback(failed)
    d.addCallback(lambda _: needsRunningReactor(reactor, reactor.stop))
    reactorBuilder.runReactor(reactor)
    return reactor