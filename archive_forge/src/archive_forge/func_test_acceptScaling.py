import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipIf(os.environ.get('INFRASTRUCTURE') == 'AZUREPIPELINES', 'Hangs on Azure Pipelines due to firewall')
def test_acceptScaling(self):
    """
        L{tcp.Port.doRead} increases the number of consecutive
        C{accept} calls it performs if all of the previous C{accept}
        calls succeed; otherwise, it reduces the number to the amount
        of successful calls.
        """
    factory = ServerFactory()
    factory.protocol = Protocol
    port = self.port(0, factory, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    clients = []

    def closeAll():
        for client in clients:
            client.close()
    self.addCleanup(closeAll)

    def connect():
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('127.0.0.1', port.getHost().port))
        return client
    clients.append(connect())
    port.numberAccepts = 1
    port.doRead()
    self.assertGreater(port.numberAccepts, 1)
    clients.append(connect())
    port.doRead()
    self.assertEqual(port.numberAccepts, 1)
    port.doRead()
    self.assertEqual(port.numberAccepts, 1)