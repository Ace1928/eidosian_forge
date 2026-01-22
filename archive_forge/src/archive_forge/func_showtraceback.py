import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
def showtraceback(self) -> None:
    """This needs to override the default traceback thing
        so it can put it into a pretty colour and maybe other
        stuff, I don't know"""
    try:
        t, v, tb = sys.exc_info()
        sys.last_type = t
        sys.last_value = v
        sys.last_traceback = tb
        tblist = traceback.extract_tb(tb)
        del tblist[:1]
        for frame in tblist:
            if self.bpython_input_re.match(frame.filename):
                frame.filename = '<input>'
        l = traceback.format_list(tblist)
        if l:
            l.insert(0, 'Traceback (most recent call last):\n')
        l[len(l):] = traceback.format_exception_only(t, v)
    finally:
        pass
    self.writetb(l)