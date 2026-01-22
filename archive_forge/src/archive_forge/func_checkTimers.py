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
def checkTimers(self):
    l1 = self.timers.values()
    l2 = list(reactor.getDelayedCalls())
    missing = []
    for dc in l1:
        if dc not in l2:
            missing.append(dc)
    if missing:
        self.finished = 1
    self.assertFalse(missing, 'Should have been missing no calls, instead ' + 'was missing ' + repr(missing))