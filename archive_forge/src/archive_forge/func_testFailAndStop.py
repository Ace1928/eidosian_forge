from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testFailAndStop(self):

    def foo(x):
        lc.stop()
        raise TestException(x)
    lc = task.LoopingCall(foo, 'bar')
    return self.assertFailure(lc.start(0.1), TestException)