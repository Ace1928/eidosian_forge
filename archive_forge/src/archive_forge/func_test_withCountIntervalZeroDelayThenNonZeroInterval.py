from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_withCountIntervalZeroDelayThenNonZeroInterval(self):
    """
        L{task.LoopingCall.withCount} with interval set to 0 will still keep
        the time when last called so when the interval is reset.
        """
    clock = task.Clock()
    deferred = defer.Deferred()
    accumulator = []
    stepsBeforeDelay = 2
    extraTimeAfterDelay = 5
    mutatedLoopInterval = 2
    durationOfDelay = 9
    skippedTime = extraTimeAfterDelay + durationOfDelay
    expectedSkipCount = skippedTime // mutatedLoopInterval
    expectedSkipCount += 1

    def foo(cnt):
        accumulator.append(cnt)
        if len(accumulator) == stepsBeforeDelay:
            return deferred
    loop = task.LoopingCall.withCount(foo)
    loop.clock = clock
    loop.start(0, now=False)
    clock.pump([1] * (stepsBeforeDelay + extraTimeAfterDelay))
    self.assertEqual([1] * stepsBeforeDelay, accumulator)
    loop.interval = mutatedLoopInterval
    deferred.callback(None)
    clock.advance(durationOfDelay)
    self.assertEqual([1, 1, expectedSkipCount], accumulator)
    clock.advance(1 * mutatedLoopInterval)
    self.assertEqual([1, 1, expectedSkipCount, 1], accumulator)
    clock.advance(2 * mutatedLoopInterval)
    self.assertEqual([1, 1, expectedSkipCount, 1, 2], accumulator)