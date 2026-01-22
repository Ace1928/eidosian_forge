from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testCursorUp(self) -> None:
    self.term.cursorUp(5)
    self.assertEqual(self.term.reportCursorPosition(), (0, 0))
    self.term.cursorDown(20)
    self.term.cursorUp(1)
    self.assertEqual(self.term.reportCursorPosition(), (0, 19))
    self.term.cursorUp(19)
    self.assertEqual(self.term.reportCursorPosition(), (0, 0))