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
def get_source_of_current_name(self) -> str:
    """Return the unicode source code of the object which is bound to the
        current name in the current input line. Throw `SourceNotFound` if the
        source cannot be found."""
    obj: Optional[Callable] = self.current_func
    try:
        if obj is None:
            line = self.current_line
            if not line.strip():
                raise SourceNotFound(_('Nothing to get source of'))
            if inspection.is_eval_safe_name(line):
                obj = self.get_object(line)
        return inspect.getsource(obj)
    except (AttributeError, NameError) as e:
        msg = _('Cannot get source: %s') % (e,)
    except OSError as e:
        msg = f'{e}'
    except TypeError as e:
        if 'built-in' in f'{e}':
            msg = _('Cannot access source of %r') % (obj,)
        else:
            msg = _('No source code found for %s') % (self.current_line,)
    raise SourceNotFound(msg)