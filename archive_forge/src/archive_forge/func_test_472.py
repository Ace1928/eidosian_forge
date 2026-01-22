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
def test_472(self):
    [self.send_key(c) for c in '(1, 2, 3)']
    with output_to_repl(self.repl):
        self.send_key('\n')
        self.send_refreshes()
        self.send_key('<UP>')
        self.repl.paint()
        [self.send_key('<LEFT>') for _ in range(4)]
        self.send_key('<BACKSPACE>')
        self.send_key('4')
        self.repl.on_enter()
        self.send_refreshes()
    screen = ['>>> (1, 2, 3)', '(1, 2, 3)', '>>> (1, 4, 3)', '(1, 4, 3)', '>>> ']
    self.assert_paint_ignoring_formatting(screen, (4, 4))