from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_noCallback(self):
    """
        The L{Deferred} returned by L{task.deferLater} fires with C{None}
        when no callback function is passed.
        """
    clock = task.Clock()
    d = task.deferLater(clock, 2.0)
    self.assertNoResult(d)
    clock.advance(2.0)
    self.assertIs(None, self.successResultOf(d))