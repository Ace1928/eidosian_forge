from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_callLaterKeepsCallsOrdered(self):
    """
        The order of calls scheduled by L{task.Clock.callLater} is honored when
        adding a new call via calling L{task.Clock.callLater} again.

        For example, if L{task.Clock.callLater} is invoked with a callable "A"
        and a time t0, and then the L{IDelayedCall} which results from that is
        C{reset} to a later time t2 which is greater than t0, and I{then}
        L{task.Clock.callLater} is invoked again with a callable "B", and time
        t1 which is less than t2 but greater than t0, "B" will be invoked before
        "A".
        """
    result = []
    expected = [('b', 2.0), ('a', 3.0)]
    clock = task.Clock()
    logtime = lambda n: result.append((n, clock.seconds()))
    call_a = clock.callLater(1.0, logtime, 'a')
    call_a.reset(3.0)
    clock.callLater(2.0, logtime, 'b')
    clock.pump([1] * 3)
    self.assertEqual(result, expected)