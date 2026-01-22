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
def test_deferredResult(self):
    """
        L{threads.deferToThreadPool} executes the function passed, and
        correctly handles the positional and keyword arguments given.
        """
    d = threads.deferToThreadPool(reactor, self.tp, lambda x, y=5: x + y, 3, y=4)
    d.addCallback(self.assertEqual, 7)
    return d