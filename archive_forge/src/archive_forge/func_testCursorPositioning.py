from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testCursorPositioning(self) -> None:
    self.term.cursorPosition(3, 9)
    self.assertEqual(self.term.reportCursorPosition(), (3, 9))