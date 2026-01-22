import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def test_internalDefGenReturnValueDoesntLeak(self):
    """
        When one inlineCallbacks calls another, the internal L{_DefGen_Return}
        flow control exception raised by calling L{defer.returnValue} doesn't
        leak into tracebacks captured in the caller.
        """
    clock = task.Clock()

    @inlineCallbacks
    def _returns():
        """
            This is the inner function using returnValue.
            """
        yield task.deferLater(clock, 0)
        returnValue('actual-value-not-used-for-the-test')

    @inlineCallbacks
    def _raises():
        try:
            yield _returns()
            raise TerminalException('boom returnValue')
        except TerminalException:
            return traceback.format_exc()
    d = _raises()
    clock.advance(0)
    tb = self.successResultOf(d)
    self.assertNotIn('_DefGen_Return', tb)
    self.assertNotIn('During handling of the above exception, another exception occurred', tb)
    self.assertIn('test_defgen.TerminalException: boom returnValue', tb)