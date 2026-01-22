import os
import sys
import time
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, threads
from twisted.python import failure, log, threadable, threadpool
from twisted.trial.unittest import TestCase
import time
import %(reactor)s
from twisted.internet import reactor
def test_asyncBlockingCallFromThread(self):
    """
        Test blockingCallFromThread as above, but be sure the resulting
        Deferred is not already fired.
        """

    def reactorFunc():
        d = defer.Deferred()
        reactor.callLater(0.1, d.callback, 'egg')
        return d

    def cb(res):
        self.assertEqual(res[0][0], 'egg')
    return self._testBlockingCallFromThread(reactorFunc).addCallback(cb)