import argparse
import collections.abc
import copy
import dataclasses
import enum
from functools import lru_cache
import glob
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import shlex
import sys
from textwrap import dedent
import types
from types import FunctionType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from pluggy import HookimplMarker
from pluggy import HookimplOpts
from pluggy import HookspecMarker
from pluggy import HookspecOpts
from pluggy import PluginManager
from .compat import PathAwareHookProxy
from .exceptions import PrintHelp as PrintHelp
from .exceptions import UsageError as UsageError
from .findpaths import determine_setup
import _pytest._code
from _pytest._code import ExceptionInfo
from _pytest._code import filter_traceback
from _pytest._io import TerminalWriter
import _pytest.deprecated
import _pytest.hookspec
from _pytest.outcomes import fail
from _pytest.outcomes import Skipped
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportMode
from _pytest.pathlib import resolve_package_path
from _pytest.pathlib import safe_exists
from _pytest.stash import Stash
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import warn_explicit_for
@lru_cache(maxsize=50)
def parse_warning_filter(arg: str, *, escape: bool) -> Tuple['warnings._ActionKind', str, Type[Warning], str, int]:
    """Parse a warnings filter string.

    This is copied from warnings._setoption with the following changes:

    * Does not apply the filter.
    * Escaping is optional.
    * Raises UsageError so we get nice error messages on failure.
    """
    __tracebackhide__ = True
    error_template = dedent(f'        while parsing the following warning configuration:\n\n          {arg}\n\n        This error occurred:\n\n        {{error}}\n        ')
    parts = arg.split(':')
    if len(parts) > 5:
        doc_url = 'https://docs.python.org/3/library/warnings.html#describing-warning-filters'
        error = dedent(f'            Too many fields ({len(parts)}), expected at most 5 separated by colons:\n\n              action:message:category:module:line\n\n            For more information please consult: {doc_url}\n            ')
        raise UsageError(error_template.format(error=error))
    while len(parts) < 5:
        parts.append('')
    action_, message, category_, module, lineno_ = (s.strip() for s in parts)
    try:
        action: 'warnings._ActionKind' = warnings._getaction(action_)
    except warnings._OptionError as e:
        raise UsageError(error_template.format(error=str(e))) from None
    try:
        category: Type[Warning] = _resolve_warning_category(category_)
    except Exception:
        exc_info = ExceptionInfo.from_current()
        exception_text = exc_info.getrepr(style='native')
        raise UsageError(error_template.format(error=exception_text)) from None
    if message and escape:
        message = re.escape(message)
    if module and escape:
        module = re.escape(module) + '\\Z'
    if lineno_:
        try:
            lineno = int(lineno_)
            if lineno < 0:
                raise ValueError('number is negative')
        except ValueError as e:
            raise UsageError(error_template.format(error=f'invalid lineno {lineno_!r}: {e}')) from None
    else:
        lineno = 0
    return (action, message, category, module, lineno)