import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
def test_delayedCallResetToLater(self):
    """
        L{DelayedCall.reset()} properly reschedules timer to later time
        """
    if hasWindowsSelectorEventLoopPolicy:
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        self.addCleanup(lambda: set_event_loop_policy(None))
    reactor = AsyncioSelectorReactor()
    timer_called_at = [None]

    def on_timer():
        timer_called_at[0] = reactor.seconds()
    start_time = reactor.seconds()
    dc = reactor.callLater(0, on_timer)
    dc.reset(0.5)
    reactor.callLater(1, reactor.stop)
    reactor.run()
    self.assertIsNotNone(timer_called_at[0])
    self.assertGreater(timer_called_at[0] - start_time, 0.4)