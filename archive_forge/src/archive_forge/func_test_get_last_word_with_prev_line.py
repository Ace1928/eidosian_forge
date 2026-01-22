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
@unittest.skip('this is the behavior of bash - not currently implemented')
def test_get_last_word_with_prev_line(self):
    self.repl.rl_history.entries = ['1', '2 3', '4 5 6']
    self.repl._set_current_line('abcde')
    self.repl.up_one_line()
    self.assertEqual(self.repl.current_line, '4 5 6')
    self.repl.get_last_word()
    self.assertEqual(self.repl.current_line, '4 5 63')
    self.repl.get_last_word()
    self.assertEqual(self.repl.current_line, '4 5 64')
    self.repl.up_one_line()
    self.assertEqual(self.repl.current_line, '2 3')