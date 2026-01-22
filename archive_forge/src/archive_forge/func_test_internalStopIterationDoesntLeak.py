import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def test_internalStopIterationDoesntLeak(self):
    """
        When one inlineCallbacks calls another, the internal L{StopIteration}
        flow control exception generated when the inner generator returns
        doesn't leak into tracebacks captured in the caller.

        This is similar to C{test_internalDefGenReturnValueDoesntLeak} but the
        inner function uses the "normal" return statemement rather than the
        C{returnValue} helper.
        """
    clock = task.Clock()

    @inlineCallbacks
    def _returns():
        yield task.deferLater(clock, 0)
        return 6

    @inlineCallbacks
    def _raises():
        try:
            yield _returns()
            raise TerminalException('boom normal return')
        except TerminalException:
            return traceback.format_exc()
    d = _raises()
    clock.advance(0)
    tb = self.successResultOf(d)
    self.assertNotIn('StopIteration', tb)
    self.assertNotIn('During handling of the above exception, another exception occurred', tb)
    self.assertIn('test_defgen.TerminalException: boom normal return', tb)