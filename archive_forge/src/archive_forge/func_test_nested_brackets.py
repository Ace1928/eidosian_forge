import os
from typing import cast
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython import config
from curtsies.window import CursorAwareWindow
def test_nested_brackets(self):
    self.process_multiple_events(['(', '[', '{'])
    self.assertEqual(self.repl._current_line, '([{')
    self.assertEqual(self.repl._cursor_offset, 3)