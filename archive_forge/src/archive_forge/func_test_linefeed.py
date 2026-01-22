from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def test_linefeed(self) -> None:
    """
        C{"
"} moves the cursor to the next row without changing the column.
        """
    self.term.cursorForward(5)
    self.assertEqual(self.term.reportCursorPosition(), (5, 0))
    self.term.insertAtCursor(b'\n')
    self.assertEqual(self.term.reportCursorPosition(), (5, 1))