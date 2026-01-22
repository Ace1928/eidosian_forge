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
def test_argspec(self):

    def foo(x, y, z=10):
        """docstring!"""
        pass
    argspec = inspection.getfuncprops('foo', foo)
    array = replpainter.formatted_argspec(argspec, 1, 30, setup_config())
    screen = [bold(cyan('foo')) + cyan(':') + cyan(' ') + cyan('(') + cyan('x') + yellow(',') + yellow(' ') + bold(cyan('y')) + yellow(',') + yellow(' ') + cyan('z') + yellow('=') + bold(cyan('10')) + yellow(')')]
    assertFSArraysEqual(fsarray(array), fsarray(screen))