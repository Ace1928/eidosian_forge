from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_withCountIntervalZero(self):
    """
        L{task.LoopingCall.withCount} with interval set to 0 calls the
        countCallable with a count of 1.
        """
    clock = task.Clock()
    accumulator = []

    def foo(cnt):
        accumulator.append(cnt)
        if len(accumulator) > 4:
            loop.stop()
    loop = task.LoopingCall.withCount(foo)
    loop.clock = clock
    deferred = loop.start(0, now=False)
    clock.pump([0] * 5)
    self.successResultOf(deferred)
    self.assertEqual([1] * 5, accumulator)