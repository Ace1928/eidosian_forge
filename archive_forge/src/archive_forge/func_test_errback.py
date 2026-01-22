from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_errback(self):
    """
        The L{Deferred} returned by L{task.deferLater} is errbacked if the
        supplied function raises an exception.
        """

    def callable():
        raise TestException()
    clock = task.Clock()
    d = task.deferLater(clock, 1, callable)
    clock.advance(1)
    return self.assertFailure(d, TestException)