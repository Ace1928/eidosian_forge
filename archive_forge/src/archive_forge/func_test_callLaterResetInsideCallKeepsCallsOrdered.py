from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_callLaterResetInsideCallKeepsCallsOrdered(self):
    """
        The order of calls scheduled by L{task.Clock.callLater} is honored when
        re-scheduling an existing call via L{IDelayedCall.reset} on the result
        of a previous call to C{callLater}, even when that call to C{reset}
        occurs within the callable scheduled by C{callLater} itself.
        """
    result = []
    expected = [('c', 3.0), ('b', 4.0)]
    clock = task.Clock()
    logtime = lambda n: result.append((n, clock.seconds()))
    call_b = clock.callLater(2.0, logtime, 'b')

    def a():
        call_b.reset(3.0)
    clock.callLater(1.0, a)
    clock.callLater(3.0, logtime, 'c')
    clock.pump([0.5] * 10)
    self.assertEqual(result, expected)