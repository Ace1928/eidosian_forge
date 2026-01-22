import contextlib
import errno
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from enum import Enum
from types import FrameType, TracebackType
from typing import (
from .._typing_compat import Literal
import greenlet
from curtsies import (
from curtsies.configfile_keynames import keymap as key_dispatch
from curtsies.input import is_main_thread
from curtsies.window import CursorAwareWindow
from cwcwidth import wcswidth
from pygments import format as pygformat
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from . import events as bpythonevents, sitefix, replpainter as paint
from ..config import Config
from .coderunner import (
from .filewatch import ModuleChangedEventHandler
from .interaction import StatusBar
from .interpreter import (
from .manual_readline import (
from .parse import parse as bpythonparse, func_for_letter, color_for_letter
from .preprocess import preprocess
from .. import __version__
from ..config import getpreferredencoding
from ..formatter import BPythonFormatter
from ..pager import get_pager_command
from ..repl import (
from ..translations import _
from ..line import CHARACTER_PAIR_MAP
def key_help_text(self) -> str:
    NOT_IMPLEMENTED = ('suspend', 'cut to buffer', 'search', 'last output', 'yank from buffer', 'cut to buffer')
    pairs = [['complete history suggestion', 'right arrow at end of line'], ['previous match with current line', 'up arrow']]
    for functionality, key in ((attr[:-4].replace('_', ' '), getattr(self.config, attr)) for attr in self.config.__dict__ if attr.endswith('key')):
        if functionality in NOT_IMPLEMENTED:
            key = 'Not Implemented'
        if key == '':
            key = 'Disabled'
        pairs.append([functionality, key])
    max_func = max((len(func) for func, key in pairs))
    return '\n'.join((f'{func.rjust(max_func)} : {key}' for func, key in pairs))