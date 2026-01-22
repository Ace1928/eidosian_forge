from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.trial.unittest import TestCase
from twisted.web import error, server
def testResults(self):
    inputOutput = [('add', (2, 3), 5), ('defer', ('a',), 'a'), ('dict', ({'a': 1}, 'a'), 1), ('triple', ('a', 1), ['a', 1, None])]
    dl = []
    for meth, args, outp in inputOutput:
        d = self.proxy().callRemote(meth, *args)
        d.addCallback(self.assertEqual, outp)
        dl.append(d)
    d = self.proxy().callRemote('complex')
    d.addCallback(lambda result: result._asdict())
    d.addCallback(self.assertEqual, {'a': ['b', 'c', 12, []], 'D': 'foo'})
    dl.append(d)
    return defer.DeferredList(dl, fireOnOneErrback=True)