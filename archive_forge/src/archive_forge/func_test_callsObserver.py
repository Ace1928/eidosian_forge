from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def test_callsObserver(self) -> None:
    """
        L{_collectWarnings} calls the observer with each emitted warning.
        """
    firstMessage = 'dummy calls observer warning'
    secondMessage = firstMessage[::-1]
    thirdMessage = Warning(1, 2, 3)
    events: list[str | _Warning] = []

    def f() -> None:
        events.append('call')
        warnings.warn(firstMessage)
        warnings.warn(secondMessage)
        warnings.warn(thirdMessage)
        events.append('returning')
    _collectWarnings(events.append, f)
    self.assertEqual(events[0], 'call')
    assert isinstance(events[1], _Warning)
    self.assertEqual(events[1].message, firstMessage)
    assert isinstance(events[2], _Warning)
    self.assertEqual(events[2].message, secondMessage)
    assert isinstance(events[3], _Warning)
    self.assertEqual(events[3].message, str(thirdMessage))
    self.assertEqual(events[4], 'returning')
    self.assertEqual(len(events), 5)