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
def current_string(self, concatenate=False):
    """If the line ends in a string get it, otherwise return ''"""
    tokens = self.tokenize(self.current_line)
    string_tokens = list(takewhile(token_is_any_of([Token.String, Token.Text]), reversed(tokens)))
    if not string_tokens:
        return ''
    opening = string_tokens.pop()[1]
    string = list()
    for token, value in reversed(string_tokens):
        if token is Token.Text:
            continue
        elif opening is None:
            opening = value
        elif token is Token.String.Doc:
            string.append(value[3:-3])
            opening = None
        elif value == opening:
            opening = None
            if not concatenate:
                string = list()
        else:
            string.append(value)
    if opening is None:
        return ''
    return ''.join(string)