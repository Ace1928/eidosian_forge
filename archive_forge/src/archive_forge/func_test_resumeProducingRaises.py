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
def test_resumeProducingRaises(self):
    """
        If the underlying producer raises an exception when resumeProducing is
        called, the streaming wrapper should log the error, unregister from
        the consumer and stop streaming.
        """
    consumer = StringTransport()
    done = self.resumeProducingRaises(consumer, [(ZeroDivisionError, 'failed, producing will be stopped')])

    def cleanShutdown(ignore):
        self.assertIsNone(consumer.producer)
    done.addCallback(cleanShutdown)
    return done