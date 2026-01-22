from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testStopRunning(self):
    """
        Test that a running iterator will not run to completion when the
        cooperator is stopped.
        """
    c = task.Cooperator()

    def myiter():
        yield from range(3)
    myiter.value = -1
    d = c.coiterate(myiter())
    d.addCallback(self.cbIter)
    d.addErrback(self.ebIter)
    c.stop()

    def doasserts(result):
        self.assertEqual(result, self.RESULT)
        self.assertEqual(myiter.value, -1)
    d.addCallback(doasserts)
    return d