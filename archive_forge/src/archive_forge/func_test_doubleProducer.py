import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_doubleProducer(self):
    """
        Verify that registering a non-streaming producer invokes its
        resumeProducing() method and that you can only register one producer
        at a time.
        """
    fd = abstract.FileDescriptor()
    fd.connected = 1
    dp = DummyProducer()
    fd.registerProducer(dp, 0)
    self.assertEqual(dp.events, ['resume'])
    self.assertRaises(RuntimeError, fd.registerProducer, DummyProducer(), 0)