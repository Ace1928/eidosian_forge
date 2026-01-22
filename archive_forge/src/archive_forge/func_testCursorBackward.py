from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testCursorBackward(self) -> None:
    self.term.cursorForward(10)
    self.term.cursorBackward(2)
    self.assertEqual(self.term.reportCursorPosition(), (8, 0))
    self.term.cursorBackward(7)
    self.assertEqual(self.term.reportCursorPosition(), (1, 0))
    self.term.cursorBackward(1)
    self.assertEqual(self.term.reportCursorPosition(), (0, 0))
    self.term.cursorBackward(1)
    self.assertEqual(self.term.reportCursorPosition(), (0, 0))