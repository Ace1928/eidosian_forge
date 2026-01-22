from __future__ import absolute_import
import linecache
import os
import platform
import sys
from dataclasses import dataclass, field
from traceback import walk_tb
from types import ModuleType, TracebackType
from typing import (
from pip._vendor.pygments.lexers import guess_lexer_for_filename
from pip._vendor.pygments.token import Comment, Keyword, Name, Number, Operator, String
from pip._vendor.pygments.token import Text as TextToken
from pip._vendor.pygments.token import Token
from pip._vendor.pygments.util import ClassNotFound
from . import pretty
from ._loop import loop_last
from .columns import Columns
from .console import Console, ConsoleOptions, ConsoleRenderable, RenderResult, group
from .constrain import Constrain
from .highlighter import RegexHighlighter, ReprHighlighter
from .panel import Panel
from .scope import render_scope
from .style import Style
from .syntax import Syntax
from .text import Text
from .theme import Theme
def ipy_display_traceback(*args: Any, is_syntax: bool=False, **kwargs: Any) -> None:
    """Internally called traceback from ip._showtraceback"""
    nonlocal tb_data
    exc_tuple = ip._get_exc_info()
    tb: Optional[TracebackType] = None if is_syntax else exc_tuple[2]
    compiled = tb_data.get('running_compiled_code', False)
    tb_offset = tb_data.get('tb_offset', 1 if compiled else 0)
    for _ in range(tb_offset):
        if tb is None:
            break
        tb = tb.tb_next
    excepthook(exc_tuple[0], exc_tuple[1], tb)
    tb_data = {}