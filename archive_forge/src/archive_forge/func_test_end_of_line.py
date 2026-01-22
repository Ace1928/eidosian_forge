import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_end_of_line(self):
    expected = (len(self._line), self._line)
    for i in range(len(self._line)):
        result = end_of_line(i, self._line)
        self.assertEqual(expected, result)