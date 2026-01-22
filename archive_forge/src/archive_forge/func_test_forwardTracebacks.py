from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_forwardTracebacks(self):
    """
        Chained inlineCallbacks are forwarding the traceback information
        from generator to generator.

        A first simple test with a couple of inline callbacks.
        """

    @inlineCallbacks
    def erroring():
        yield 'forcing generator'
        raise Exception('Error Marker')

    @inlineCallbacks
    def calling():
        yield erroring()
    d = calling()
    f = self.failureResultOf(d)
    tb = f.getTraceback()
    self.assertIn('in erroring', tb)
    self.assertIn('in calling', tb)
    self.assertIn('Error Marker', tb)