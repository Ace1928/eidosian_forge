import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_backspace(self):
    self.assertEqual(backspace(2, 'as'), (1, 'a'))
    self.assertEqual(backspace(3, 'as '), (2, 'as'))