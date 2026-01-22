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
def test_stopProducing(self):
    """
        When the streaming producer is stopped by the consumer, the underlying
        producer is stopped, and streaming is stopped.
        """

    class StoppingStringTransport(StringTransport):
        writes = 0

        def write(self, data):
            self.writes += 1
            StringTransport.write(self, data)
            if self.writes == 3:
                self.producer.stopProducing()
    consumer = StoppingStringTransport()
    nsProducer = NonStreamingProducer(consumer)
    streamingProducer = _PullToPush(nsProducer, consumer)
    consumer.registerProducer(streamingProducer, True)
    done = nsProducer.result

    def doneStreaming(_):
        self.assertEqual(consumer.value(), b'012')
        self.assertTrue(nsProducer.stopped)
        self.assertTrue(streamingProducer._finished)
    done.addCallback(doneStreaming)
    streamingProducer.startStreaming()
    return done