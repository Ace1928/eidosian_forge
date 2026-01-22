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
@hookimpl(trylast=True)
def pytest_load_initial_conftests(self, early_config: 'Config') -> None:
    args, args_source = early_config._decide_args(args=early_config.known_args_namespace.file_or_dir, pyargs=early_config.known_args_namespace.pyargs, testpaths=early_config.getini('testpaths'), invocation_dir=early_config.invocation_params.dir, rootpath=early_config.rootpath, warn=False)
    self.pluginmanager._set_initial_conftests(args=args, pyargs=early_config.known_args_namespace.pyargs, noconftest=early_config.known_args_namespace.noconftest, rootpath=early_config.rootpath, confcutdir=early_config.known_args_namespace.confcutdir, invocation_dir=early_config.invocation_params.dir, importmode=early_config.known_args_namespace.importmode, consider_namespace_packages=early_config.getini('consider_namespace_packages'))