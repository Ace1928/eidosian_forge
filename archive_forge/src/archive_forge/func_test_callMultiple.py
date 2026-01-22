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
def test_callMultiple(self):
    """
        L{threads.callMultipleInThread} calls multiple functions in a thread.
        """
    L = []
    N = 10
    d = defer.Deferred()

    def finished():
        self.assertEqual(L, list(range(N)))
        d.callback(None)
    threads.callMultipleInThread([(L.append, (i,), {}) for i in range(N)] + [(reactor.callFromThread, (finished,), {})])
    return d