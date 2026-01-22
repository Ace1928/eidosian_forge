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
def test_tab_with_no_matches_selects_first(self):
    self.repl._current_line = ' aa'
    self.repl._cursor_offset = 3
    self.repl.matches_iter.matches = []
    self.repl.matches_iter.is_cseq.return_value = False
    mock_next(self.repl.matches_iter, None)
    self.repl.matches_iter.cur_line.return_value = (None, None)
    self.repl.on_tab()
    self.repl.complete.assert_called_once_with(tab=True)
    self.repl.matches_iter.cur_line.assert_called_once_with()