import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def testBrokenTimer(self):
    d = defer.Deferred()
    t = internet.TimerService(1, lambda: 1 // 0)
    oldFailed = t._failed

    def _failed(why):
        oldFailed(why)
        d.callback(None)
    t._failed = _failed
    t.startService()
    d.addCallback(lambda x: t.stopService)
    d.addCallback(lambda x: self.assertEqual([ZeroDivisionError], [o.value.__class__ for o in self.flushLoggedErrors(ZeroDivisionError)]))
    return d