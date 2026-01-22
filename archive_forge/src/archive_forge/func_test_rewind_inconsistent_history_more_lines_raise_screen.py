import itertools
import os
import pydoc
import string
import sys
from contextlib import contextmanager
from typing import cast
from curtsies.formatstringarray import (
from curtsies.fmtfuncs import cyan, bold, green, yellow, on_magenta, red
from curtsies.window import CursorAwareWindow
from unittest import mock, skipIf
from bpython.curtsiesfrontend.events import RefreshRequestEvent
from bpython import config, inspection
from bpython.curtsiesfrontend.repl import BaseRepl
from bpython.curtsiesfrontend import replpainter
from bpython.curtsiesfrontend.repl import (
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
def test_rewind_inconsistent_history_more_lines_raise_screen(self):
    self.repl.width = 60
    sys.a = 5
    self.enter('import sys')
    self.enter('for i in range(sys.a):')
    self.enter('    print(sys.a)')
    self.enter('')
    self.enter('1 + 1')
    self.enter('2 + 2')
    screen = ['>>> import sys', '>>> for i in range(sys.a):', '...     print(sys.a)', '... ', '5', '5', '5', '5', '5', '>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
    self.assert_paint_ignoring_formatting(screen, (13, 4))
    self.repl.scroll_offset += len(screen) - self.repl.height
    self.assert_paint_ignoring_formatting(screen[9:], (4, 4))
    sys.a = 1
    self.undo()
    screen = [INCONSISTENT_HISTORY_MSG[:self.repl.width], '1', '>>> 1 + 1', '2', '>>> ', ' ']
    self.assert_paint_ignoring_formatting(screen)
    self.repl.scroll_offset += len(screen) - self.repl.height
    self.assert_paint_ignoring_formatting(screen[1:-1])