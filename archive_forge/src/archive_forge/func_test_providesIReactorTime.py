from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_providesIReactorTime(self):
    c = task.Clock()
    self.assertTrue(interfaces.IReactorTime.providedBy(c), 'Clock does not provide IReactorTime')