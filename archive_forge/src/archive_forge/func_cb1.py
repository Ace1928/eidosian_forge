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
def cb1(ign):

    def threadedFunc():
        try:
            r = threads.blockingCallFromThread(reactor, reactorFunc)
        except Exception as e:
            errors.append(e)
        else:
            results.append(r)
        waiter.set()
    reactor.callInThread(threadedFunc)
    return threads.deferToThread(waiter.wait, self.getTimeout())