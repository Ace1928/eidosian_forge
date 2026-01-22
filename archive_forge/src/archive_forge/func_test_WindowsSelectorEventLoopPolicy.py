import gc
import sys
from asyncio import (
from unittest import skipIf
from twisted.internet.asyncioreactor import AsyncioSelectorReactor
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
@skipIf(not hasWindowsSelectorEventLoopPolicy, 'WindowsSelectorEventLoopPolicy only on Windows')
def test_WindowsSelectorEventLoopPolicy(self):
    """
        L{AsyncioSelectorReactor} will work if
        if L{asyncio.WindowsSelectorEventLoopPolicy} is default.
        """
    set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    self.addCleanup(lambda: set_event_loop_policy(None))
    reactor = AsyncioSelectorReactor()
    self.assertReactorWorksWithAsyncioFuture(reactor)