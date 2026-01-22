import unittest as pyunit
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
def test_assertFailure_wrongException(self):
    d = defer.maybeDeferred(lambda: 1 / 0)
    self.assertFailure(d, OverflowError)
    d.addCallbacks(lambda x: self.fail('Should have failed'), lambda x: x.trap(self.failureException))
    return d