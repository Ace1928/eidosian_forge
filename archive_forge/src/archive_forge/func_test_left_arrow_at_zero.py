import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_left_arrow_at_zero(self):
    pos = 0
    expected = (pos, self._line)
    result = left_arrow(pos, self._line)
    self.assertEqual(expected, result)