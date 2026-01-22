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
def test_callLater(self):
    """
        Test that a DelayedCall really calls the function it is
        supposed to call.
        """
    d = Deferred()
    reactor.callLater(0, d.callback, None)
    d.addCallback(self.assertEqual, None)
    return d