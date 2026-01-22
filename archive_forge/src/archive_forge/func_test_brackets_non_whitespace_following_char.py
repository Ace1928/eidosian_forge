import os
from typing import cast
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython import config
from curtsies.window import CursorAwareWindow
def test_brackets_non_whitespace_following_char(self):
    self.repl.current_line = "s = s.connect('localhost', 8080)"
    self.repl.cursor_offset = 14
    self.repl.process_event('(')
    self.assertEqual(self.repl._current_line, "s = s.connect(('localhost', 8080)")
    self.assertEqual(self.repl._cursor_offset, 15)