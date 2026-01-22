from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def testOvertype(self) -> None:
    s = b'hello, world.'
    self.term.write(s)
    self.term.cursorBackward(len(s))
    self.term.resetModes([modes.IRM])
    self.term.write(b'H')
    self.assertEqual(self.term.__bytes__(), b'H' + s[1:] + b'\n' + b'\n' * (HEIGHT - 2))