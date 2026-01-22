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
def test_unconnectedFileDescriptor(self):
    """
        Verify that registering a producer when the connection has already
        been closed invokes its stopProducing() method.
        """
    fd = abstract.FileDescriptor()
    fd.disconnected = 1
    dp = DummyProducer()
    fd.registerProducer(dp, 0)
    self.assertEqual(dp.events, ['stop'])