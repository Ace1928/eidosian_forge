from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def test_resetPrivateModes(self) -> None:
    """
        Verify that L{helper.TerminalBuffer.resetPrivateModes} changes the Set
        Mode (SM) state to "reset" for the private modes it is passed.
        """
    expected = self.term.privateModes.copy()
    self.term.resetPrivateModes([privateModes.AUTO_WRAP, privateModes.CURSOR_MODE])
    del expected[privateModes.AUTO_WRAP]
    del expected[privateModes.CURSOR_MODE]
    self.assertEqual(expected, self.term.privateModes)