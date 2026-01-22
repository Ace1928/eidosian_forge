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
def completion_target(num_names, chars_in_first_name=1):

    class Class:
        pass
    if chars_in_first_name < 1:
        raise ValueError('need at least one char in each name')
    elif chars_in_first_name == 1 and num_names > len(string.ascii_letters):
        raise ValueError('need more chars to make so many names')
    names = gen_names()
    if num_names > 0:
        setattr(Class, 'a' * chars_in_first_name, 1)
        next(names)
    for _, name in zip(range(num_names - 1), names):
        setattr(Class, name, 0)
    return Class()