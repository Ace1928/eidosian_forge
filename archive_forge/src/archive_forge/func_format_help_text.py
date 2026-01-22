import enum
import errno
import inspect
import os
import sys
import typing as t
from collections import abc
from contextlib import contextmanager
from contextlib import ExitStack
from functools import update_wrapper
from gettext import gettext as _
from gettext import ngettext
from itertools import repeat
from types import TracebackType
from . import types
from .exceptions import Abort
from .exceptions import BadParameter
from .exceptions import ClickException
from .exceptions import Exit
from .exceptions import MissingParameter
from .exceptions import UsageError
from .formatting import HelpFormatter
from .formatting import join_options
from .globals import pop_context
from .globals import push_context
from .parser import _flag_needs_value
from .parser import OptionParser
from .parser import split_opt
from .termui import confirm
from .termui import prompt
from .termui import style
from .utils import _detect_program_name
from .utils import _expand_args
from .utils import echo
from .utils import make_default_short_help
from .utils import make_str
from .utils import PacifyFlushWrapper
def format_help_text(self, ctx: Context, formatter: HelpFormatter) -> None:
    """Writes the help text to the formatter if it exists."""
    if self.help is not None:
        text = inspect.cleandoc(self.help).partition('\x0c')[0]
    else:
        text = ''
    if self.deprecated:
        text = _('(Deprecated) {text}').format(text=text)
    if text:
        formatter.write_paragraph()
        with formatter.indentation():
            formatter.write_text(text)