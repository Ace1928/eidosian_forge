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
def issue_config_time_warning(self, warning: Warning, stacklevel: int) -> None:
    """Issue and handle a warning during the "configure" stage.

        During ``pytest_configure`` we can't capture warnings using the ``catch_warnings_for_item``
        function because it is not possible to have hook wrappers around ``pytest_configure``.

        This function is mainly intended for plugins that need to issue warnings during
        ``pytest_configure`` (or similar stages).

        :param warning: The warning instance.
        :param stacklevel: stacklevel forwarded to warnings.warn.
        """
    if self.pluginmanager.is_blocked('warnings'):
        return
    cmdline_filters = self.known_args_namespace.pythonwarnings or []
    config_filters = self.getini('filterwarnings')
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always', type(warning))
        apply_warning_filters(config_filters, cmdline_filters)
        warnings.warn(warning, stacklevel=stacklevel)
    if records:
        frame = sys._getframe(stacklevel - 1)
        location = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
        self.hook.pytest_warning_recorded.call_historic(kwargs=dict(warning_message=records[0], when='config', nodeid='', location=location))