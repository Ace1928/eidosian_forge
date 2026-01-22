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
def threadedFunction():
    for i in range(100000):
        try:
            reactor.callFromThread(lambda: None)
        except BaseException:
            self.failure = failure.Failure()
            break
    waiter.set()