import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_serverReadFailure(self):
    """
        When a server fails to successfully read a packet the server should
        still be able to process future packets.
        The IOCP reactor had a historical problem where a failure to read caused
        the reactor to ignore any future reads. This test should prevent a regression.

        Note: This test assumes no one is listening on port 80 UDP.
        """
    client = GoodClient()
    clientStarted = client.startedDeferred = defer.Deferred()
    clientPort = reactor.listenUDP(0, client, interface='127.0.0.1')
    test_data_to_send = b'Sending test packet to server'
    server = Server()
    serverStarted = server.startedDeferred = defer.Deferred()
    serverGotData = server.packetReceived = defer.Deferred()
    serverPort = reactor.listenUDP(0, server, interface='127.0.0.1')
    server_client_started_d = defer.DeferredList([clientStarted, serverStarted], fireOnOneErrback=True)

    def cbClientAndServerStarted(ignored):
        server.transport.write(b'write to port no one is listening to', ('127.0.0.1', 80))
        client.transport.write(test_data_to_send, ('127.0.0.1', serverPort._realPortNumber))
    server_client_started_d.addCallback(cbClientAndServerStarted)
    all_data_sent = defer.DeferredList([server_client_started_d, serverGotData], fireOnOneErrback=True)

    def verify_server_got_data(ignored):
        self.assertEqual(server.packets[0][0], test_data_to_send)
    all_data_sent.addCallback(verify_server_got_data)

    def cleanup(ignored):
        return defer.DeferredList([defer.maybeDeferred(clientPort.stopListening), defer.maybeDeferred(serverPort.stopListening)], fireOnOneErrback=True)
    all_data_sent.addCallback(cleanup)
    return all_data_sent