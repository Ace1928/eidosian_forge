from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_formatDelayedCallsAndSelectables(self) -> None:
    """
        Both delayed calls and selectables can appear in the same error.
        """
    error = DirtyReactorAggregateError(['bleck', 'Boozo'], ['Sel1', 'Sel2'])
    self.assertEqual(str(error), 'Reactor was unclean.\nDelayedCalls: (set twisted.internet.base.DelayedCall.debug = True to debug)\nbleck\nBoozo\nSelectables:\nSel1\nSel2')