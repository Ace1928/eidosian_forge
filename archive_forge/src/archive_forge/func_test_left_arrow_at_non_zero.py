import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_left_arrow_at_non_zero(self):
    for i in range(1, len(self._line)):
        expected = (i - 1, self._line)
        result = left_arrow(i, self._line)
        self.assertEqual(expected, result)