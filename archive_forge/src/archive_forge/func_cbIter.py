from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def cbIter(self, ign):
    self.fail()