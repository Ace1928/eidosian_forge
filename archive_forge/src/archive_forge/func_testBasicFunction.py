from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testBasicFunction(self):
    timings = [0.05, 0.1, 0.1]
    clock = task.Clock()
    L = []

    def foo(a, b, c=None, d=None):
        L.append((a, b, c, d))
    lc = TestableLoopingCall(clock, foo, 'a', 'b', d='d')
    D = lc.start(0.1)
    theResult = []

    def saveResult(result):
        theResult.append(result)
    D.addCallback(saveResult)
    clock.pump(timings)
    self.assertEqual(len(L), 3, 'got %d iterations, not 3' % (len(L),))
    for a, b, c, d in L:
        self.assertEqual(a, 'a')
        self.assertEqual(b, 'b')
        self.assertIsNone(c)
        self.assertEqual(d, 'd')
    lc.stop()
    self.assertIs(theResult[0], lc)
    self.assertFalse(clock.calls)