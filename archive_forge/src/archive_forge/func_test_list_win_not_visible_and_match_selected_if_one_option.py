import code
import os
import sys
import tempfile
import io
from typing import cast
import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython.curtsiesfrontend import interpreter
from bpython.curtsiesfrontend import events as bpythonevents
from bpython.curtsiesfrontend.repl import LineType
from bpython import autocomplete
from bpython import config
from bpython import args
from bpython.test import (
from curtsies import events
from curtsies.window import CursorAwareWindow
from importlib import invalidate_caches
def test_list_win_not_visible_and_match_selected_if_one_option(self):
    self.repl._current_line = " './a'"
    self.repl._cursor_offset = 5
    with mock.patch('bpython.autocomplete.get_completer') as m:
        m.return_value = (['./abcd'], autocomplete.FilenameCompletion())
        self.repl.update_completion()
        self.assertEqual(self.repl.list_win_visible, False)
    self.repl.on_tab()
    self.assertEqual(self.repl._current_line, " './abcd'")
    self.assertEqual(self.repl.current_match, None)
    self.assertEqual(self.repl.list_win_visible, False)