from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_reactorTimeSkips(self):
    """
        When more time than the defined interval passes between when
        L{LoopingCall} schedules itself to run again and when it actually
        runs again, it should schedule the next call for the next interval
        which is still in the future.
        """
    times = []
    clock = task.Clock()

    def aCallback():
        times.append(clock.seconds())
    call = task.LoopingCall(aCallback)
    call.clock = clock
    call.start(0.5)
    self.assertEqual(times, [0])
    clock.advance(2)
    self.assertEqual(times, [0, 2])
    clock.advance(1)
    self.assertEqual(times, [0, 2, 3])
    clock.advance(0)
    self.assertEqual(times, [0, 2, 3])