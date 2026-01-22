from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testUnexpectedErrorNotActuallyLater(self):

    def myiter():
        yield defer.fail(RuntimeError())
    c = task.Cooperator()
    d = c.coiterate(myiter())
    return self.assertFailure(d, RuntimeError)