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
def resumeProducingRaises(self, consumer, expectedExceptions):
    """
        Common implementation for tests where the underlying producer throws
        an exception when its resumeProducing is called.
        """

    class ThrowingProducer(NonStreamingProducer):

        def resumeProducing(self):
            if self.counter == 2:
                return 1 / 0
            else:
                NonStreamingProducer.resumeProducing(self)
    nsProducer = ThrowingProducer(consumer)
    streamingProducer = _PullToPush(nsProducer, consumer)
    consumer.registerProducer(streamingProducer, True)
    loggedMsgs = []
    log.addObserver(loggedMsgs.append)
    self.addCleanup(log.removeObserver, loggedMsgs.append)

    def unregister(orig=consumer.unregisterProducer):
        orig()
        streamingProducer.stopStreaming()
    consumer.unregisterProducer = unregister
    streamingProducer.startStreaming()
    done = streamingProducer._coopTask.whenDone()
    done.addErrback(lambda reason: reason.trap(TaskStopped))

    def stopped(ign):
        self.assertEqual(consumer.value(), b'01')
        errors = self.flushLoggedErrors()
        self.assertEqual(len(errors), len(expectedExceptions))
        for f, (expected, msg), logMsg in zip(errors, expectedExceptions, loggedMsgs):
            self.assertTrue(f.check(expected))
            self.assertIn(msg, logMsg['why'])
        self.assertTrue(streamingProducer._finished)
    done.addCallback(stopped)
    return done