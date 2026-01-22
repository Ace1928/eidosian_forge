from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testUnexpectedError(self):
    c = task.Cooperator()

    def myiter():
        if False:
            yield None
        else:
            raise RuntimeError()
    d = c.coiterate(myiter())
    return self.assertFailure(d, RuntimeError)