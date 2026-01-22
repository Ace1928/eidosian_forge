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
def test_wakerOverflow(self):
    """
        Try to make an overflow on the reactor waker using callFromThread.
        """

    def cb(ign):
        self.failure = None
        waiter = threading.Event()

        def threadedFunction():
            for i in range(100000):
                try:
                    reactor.callFromThread(lambda: None)
                except BaseException:
                    self.failure = failure.Failure()
                    break
            waiter.set()
        reactor.callInThread(threadedFunction)
        waiter.wait(120)
        if not waiter.isSet():
            self.fail('Timed out waiting for event')
        if self.failure is not None:
            return defer.fail(self.failure)
    return self._waitForThread().addCallback(cb)