from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_reactorTimeCountSkips(self):
    """
        When L{LoopingCall} schedules itself to run again, if more than the
        specified interval has passed, it should schedule the next call for the
        next interval which is still in the future. If it was created
        using L{LoopingCall.withCount}, a positional argument will be
        inserted at the beginning of the argument list, indicating the number
        of calls that should have been made.
        """
    times = []
    clock = task.Clock()

    def aCallback(numCalls):
        times.append((clock.seconds(), numCalls))
    call = task.LoopingCall.withCount(aCallback)
    call.clock = clock
    INTERVAL = 0.5
    REALISTIC_DELAY = 0.01
    call.start(INTERVAL)
    self.assertEqual(times, [(0, 1)])
    clock.advance(INTERVAL + REALISTIC_DELAY)
    self.assertEqual(times, [(0, 1), (INTERVAL + REALISTIC_DELAY, 1)])
    clock.advance(3 * INTERVAL + REALISTIC_DELAY)
    self.assertEqual(times, [(0, 1), (INTERVAL + REALISTIC_DELAY, 1), (4 * INTERVAL + 2 * REALISTIC_DELAY, 3)])
    clock.advance(0)
    self.assertEqual(times, [(0, 1), (INTERVAL + REALISTIC_DELAY, 1), (4 * INTERVAL + 2 * REALISTIC_DELAY, 3)])