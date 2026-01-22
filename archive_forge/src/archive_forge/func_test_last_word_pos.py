import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_last_word_pos(self):
    line = 'a word'
    expected = 2
    result = last_word_pos(line)
    self.assertEqual(expected, result)