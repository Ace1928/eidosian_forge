from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_returnValueNonLocalWarning(self):
    """
        L{returnValue} will emit a non-local exit warning in the simplest case,
        where the offending function is invoked immediately.
        """

    @inlineCallbacks
    def inline():
        self.mistakenMethod()
        returnValue(2)
        yield 0
    d = inline()
    results = []
    d.addCallback(results.append)
    self.assertMistakenMethodWarning(results)