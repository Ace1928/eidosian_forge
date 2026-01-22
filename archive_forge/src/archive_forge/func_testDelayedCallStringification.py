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
def testDelayedCallStringification(self):
    dc = reactor.callLater(0, lambda x, y: None, 'x', y=10)
    str(dc)
    dc.reset(5)
    str(dc)
    dc.cancel()
    str(dc)
    dc = reactor.callLater(0, lambda: None, *range(10), x=[({'hello': 'world'}, 10j), reactor])
    str(dc)
    dc.cancel()
    str(dc)

    def calledBack(ignored):
        str(dc)
    d = Deferred().addCallback(calledBack)
    dc = reactor.callLater(0, d.callback, None)
    str(dc)
    return d