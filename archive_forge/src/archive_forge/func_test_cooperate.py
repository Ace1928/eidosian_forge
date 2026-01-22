from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def test_cooperate(self):
    """
        L{twisted.internet.task.cooperate} ought to run the generator that it is
        """
    d = defer.Deferred()

    def doit():
        yield 1
        yield 2
        yield 3
        d.callback('yay')
    it = doit()
    theTask = task.cooperate(it)
    self.assertIn(theTask, task._theCooperator._tasks)
    return d