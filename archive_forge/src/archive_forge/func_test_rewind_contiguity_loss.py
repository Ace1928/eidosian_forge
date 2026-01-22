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
def test_rewind_contiguity_loss(self):
    self.enter('1 + 1')
    self.enter('2 + 2')
    self.enter('def foo(x):')
    self.repl.current_line = '    return x + 1'
    screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> def foo(x):', '...     return x + 1']
    self.assert_paint_ignoring_formatting(screen, (5, 8))
    self.repl.scroll_offset = 1
    self.assert_paint_ignoring_formatting(screen[1:], (4, 8))
    self.undo()
    screen = ['2', '>>> 2 + 2', '4', '>>> ']
    self.assert_paint_ignoring_formatting(screen, (3, 4))
    self.undo()
    screen = ['2', '>>> ']
    self.assert_paint_ignoring_formatting(screen, (1, 4))
    self.undo()
    screen = [CONTIGUITY_BROKEN_MSG[:self.repl.width], '>>> ', '', '', '', ' ']
    self.assert_paint_ignoring_formatting(screen, (1, 4))
    screen = ['>>> ']
    self.assert_paint_ignoring_formatting(screen, (0, 4))