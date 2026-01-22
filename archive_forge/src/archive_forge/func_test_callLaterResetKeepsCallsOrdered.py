from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_callLaterResetKeepsCallsOrdered(self):
    """
        The order of calls scheduled by L{task.Clock.callLater} is honored when
        re-scheduling an existing call via L{IDelayedCall.reset} on the result
        of a previous call to C{callLater}.

        For example, if L{task.Clock.callLater} is invoked with a callable "A"
        and a time t0, and then L{task.Clock.callLater} is invoked again with a
        callable "B", and time t1 greater than t0, and finally the
        L{IDelayedCall} for "A" is C{reset} to a later time, t2, which is
        greater than t1, "B" will be invoked before "A".
        """
    result = []
    expected = [('b', 2.0), ('a', 3.0)]
    clock = task.Clock()
    logtime = lambda n: result.append((n, clock.seconds()))
    call_a = clock.callLater(1.0, logtime, 'a')
    clock.callLater(2.0, logtime, 'b')
    call_a.reset(3.0)
    clock.pump([1] * 3)
    self.assertEqual(result, expected)