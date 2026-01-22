import ast
import dataclasses
import inspect
from inspect import CO_VARARGS
from inspect import CO_VARKEYWORDS
from io import StringIO
import os
from pathlib import Path
import re
import sys
import traceback
from traceback import format_exception_only
from types import CodeType
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import final
from typing import Generic
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import SupportsIndex
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import pluggy
import _pytest
from _pytest._code.source import findsource
from _pytest._code.source import getrawcode
from _pytest._code.source import getstatementrange_ast
from _pytest._code.source import Source
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import safeformat
from _pytest._io.saferepr import saferepr
from _pytest.compat import get_real_func
from _pytest.deprecated import check_ispytest
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
def repr_traceback(self, excinfo: ExceptionInfo[BaseException]) -> 'ReprTraceback':
    traceback = excinfo.traceback
    if callable(self.tbfilter):
        traceback = self.tbfilter(excinfo)
    elif self.tbfilter:
        traceback = traceback.filter(excinfo)
    if isinstance(excinfo.value, RecursionError):
        traceback, extraline = self._truncate_recursive_traceback(traceback)
    else:
        extraline = None
    if not traceback:
        if extraline is None:
            extraline = 'All traceback entries are hidden. Pass `--full-trace` to see hidden and internal frames.'
        entries = [self.repr_traceback_entry(None, excinfo)]
        return ReprTraceback(entries, extraline, style=self.style)
    last = traceback[-1]
    if self.style == 'value':
        entries = [self.repr_traceback_entry(last, excinfo)]
        return ReprTraceback(entries, None, style=self.style)
    entries = [self.repr_traceback_entry(entry, excinfo if last == entry else None) for entry in traceback]
    return ReprTraceback(entries, extraline, style=self.style)