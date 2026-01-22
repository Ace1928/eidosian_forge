from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testCooperation(self):
    L = []

    def myiter(things):
        for th in things:
            L.append(th)
            yield None
    groupsOfThings = ['abc', (1, 2, 3), 'def', (4, 5, 6)]
    c = task.Cooperator()
    tasks = []
    for stuff in groupsOfThings:
        tasks.append(c.coiterate(myiter(stuff)))
    return defer.DeferredList(tasks).addCallback(lambda ign: self.assertEqual(tuple(L), sum(zip(*groupsOfThings), ())))