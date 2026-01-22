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
def test_streamingProducerResumedInWriteBlockedOnReadMode(self):
    """
        When the TLS transport is blocked on reads, it correctly calls
        resumeProducing on the registered producer.
        """
    clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
    clientProtocol.transport.write(b'hello world' * 320000)
    self.assertEqual(producer.producerHistory, ['pause'])
    serverProtocol, serverTLSProtocol = buildTLSProtocol(server=True)
    self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
    self.assertEqual(producer.producerHistory, ['pause', 'resume'])
    self.assertFalse(tlsProtocol._producer._producerPaused)
    self.assertFalse(tlsProtocol.transport.disconnecting)
    self.assertEqual(producer.producerState, 'producing')