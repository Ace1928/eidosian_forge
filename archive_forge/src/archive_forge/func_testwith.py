from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testwith(stuff):
    c = task.Cooperator()
    c.stop()
    d = c.coiterate(iter(()), stuff)
    d.addCallback(self.cbIter)
    d.addErrback(self.ebIter)
    return d.addCallback(lambda result: self.assertEqual(result, self.RESULT))