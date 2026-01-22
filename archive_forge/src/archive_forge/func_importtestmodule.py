import abc
from collections import Counter
from collections import defaultdict
import dataclasses
import enum
import fnmatch
from functools import partial
import inspect
import itertools
import os
from pathlib import Path
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import _pytest
from _pytest import fixtures
from _pytest import nodes
from _pytest._code import filter_traceback
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest._code.code import Traceback
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import saferepr
from _pytest.compat import ascii_escaped
from _pytest.compat import get_default_arg_names
from _pytest.compat import get_real_func
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_async_function
from _pytest.compat import is_generator
from _pytest.compat import LEGACY_PATH
from _pytest.compat import NOTSET
from _pytest.compat import safe_getattr
from _pytest.compat import safe_isclass
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import FixtureRequest
from _pytest.fixtures import FuncFixtureInfo
from _pytest.fixtures import get_scope_node
from _pytest.main import Session
from _pytest.mark import MARK_GEN
from _pytest.mark import ParameterSet
from _pytest.mark.structures import get_unpacked_marks
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import normalize_mark_list
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportPathMismatchError
from _pytest.pathlib import scandir
from _pytest.scope import _ScopeName
from _pytest.scope import Scope
from _pytest.stash import StashKey
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestReturnNotNoneWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
def importtestmodule(path: Path, config: Config):
    importmode = config.getoption('--import-mode')
    try:
        mod = import_path(path, mode=importmode, root=config.rootpath, consider_namespace_packages=config.getini('consider_namespace_packages'))
    except SyntaxError as e:
        raise nodes.Collector.CollectError(ExceptionInfo.from_current().getrepr(style='short')) from e
    except ImportPathMismatchError as e:
        raise nodes.Collector.CollectError('import file mismatch:\nimported module {!r} has this __file__ attribute:\n  {}\nwhich is not the same as the test file we want to collect:\n  {}\nHINT: remove __pycache__ / .pyc files and/or use a unique basename for your test file modules'.format(*e.args)) from e
    except ImportError as e:
        exc_info = ExceptionInfo.from_current()
        if config.getoption('verbose') < 2:
            exc_info.traceback = exc_info.traceback.filter(filter_traceback)
        exc_repr = exc_info.getrepr(style='short') if exc_info.traceback else exc_info.exconly()
        formatted_tb = str(exc_repr)
        raise nodes.Collector.CollectError(f"ImportError while importing test module '{path}'.\nHint: make sure your test modules/packages have valid Python names.\nTraceback:\n{formatted_tb}") from e
    except skip.Exception as e:
        if e.allow_module_level:
            raise
        raise nodes.Collector.CollectError("Using pytest.skip outside of a test will skip the entire module. If that's your intention, pass `allow_module_level=True`. If you want to skip a specific test or an entire class, use the @pytest.mark.skip or @pytest.mark.skipif decorators.") from e
    config.pluginmanager.consider_module(mod)
    return mod