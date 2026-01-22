from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
def prepare_header(self, etype: str, long_version: bool=False):
    colors = self.Colors
    colorsnormal = colors.Normal
    exc = '%s%s%s' % (colors.excName, etype, colorsnormal)
    width = min(75, get_terminal_size()[0])
    if long_version:
        pyver = 'Python ' + sys.version.split()[0] + ': ' + sys.executable
        date = time.ctime(time.time())
        head = '%s%s%s\n%s%s%s\n%s' % (colors.topline, '-' * width, colorsnormal, exc, ' ' * (width - len(etype) - len(pyver)), pyver, date.rjust(width))
        head += '\nA problem occurred executing Python code.  Here is the sequence of function\ncalls leading up to the error, with the most recent (innermost) call last.'
    else:
        head = '%s%s' % (exc, 'Traceback (most recent call last)'.rjust(width - len(etype)))
    return head