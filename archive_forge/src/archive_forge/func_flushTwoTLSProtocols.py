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
def flushTwoTLSProtocols(self, tlsProtocol, serverTLSProtocol):
    """
        Transfer bytes back and forth between two TLS protocols.
        """
    for i in range(3):
        clientData = self.drain(tlsProtocol.transport, True)
        if clientData:
            serverTLSProtocol.dataReceived(clientData)
        serverData = self.drain(serverTLSProtocol.transport, True)
        if serverData:
            tlsProtocol.dataReceived(serverData)
        if not serverData and (not clientData):
            break
    self.assertEqual(tlsProtocol.transport.value(), b'')
    self.assertEqual(serverTLSProtocol.transport.value(), b'')