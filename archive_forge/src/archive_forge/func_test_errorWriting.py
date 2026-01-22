from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_errorWriting(self):
    """
        Errors while writing cause the protocols to be disconnected.
        """
    tlsClient, tlsServer, handshakeDeferred, disconnectDeferred = self.handshakeProtocols()
    reason = []
    tlsClient.wrappedProtocol.connectionLost = reason.append

    class Wrapper:

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, attr):
            return getattr(self._wrapped, attr)

        def send(self, *args):
            raise Error([('SSL routines', '', 'this message is probably useless')])
    tlsClient._tlsConnection = Wrapper(tlsClient._tlsConnection)

    def handshakeDone(ign):
        tlsClient.write(b'hello')
    handshakeDeferred.addCallback(handshakeDone)

    def disconnected(ign):
        self.assertTrue(reason[0].check(Error), reason[0])
    disconnectDeferred.addCallback(disconnected)
    return disconnectDeferred