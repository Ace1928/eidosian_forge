import collections.abc
import contextlib
from fnmatch import fnmatch
import gc
import importlib
from io import StringIO
import locale
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from weakref import WeakKeyDictionary
from iniconfig import IniConfig
from iniconfig import SectionWrapper
from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import make_numbered_dir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestWarning
def inline_run(self, *args: Union[str, 'os.PathLike[str]'], plugins=(), no_reraise_ctrlc: bool=False) -> HookRecorder:
    """Run ``pytest.main()`` in-process, returning a HookRecorder.

        Runs the :py:func:`pytest.main` function to run all of pytest inside
        the test process itself.  This means it can return a
        :py:class:`HookRecorder` instance which gives more detailed results
        from that run than can be done by matching stdout/stderr from
        :py:meth:`runpytest`.

        :param args:
            Command line arguments to pass to :py:func:`pytest.main`.
        :param plugins:
            Extra plugin instances the ``pytest.main()`` instance should use.
        :param no_reraise_ctrlc:
            Typically we reraise keyboard interrupts from the child run. If
            True, the KeyboardInterrupt exception is captured.
        """
    importlib.invalidate_caches()
    plugins = list(plugins)
    finalizers = []
    try:
        finalizers.append(self.__take_sys_modules_snapshot().restore)
        finalizers.append(SysPathsSnapshot().restore)
        rec = []

        class Collect:

            def pytest_configure(x, config: Config) -> None:
                rec.append(self.make_hook_recorder(config.pluginmanager))
        plugins.append(Collect())
        ret = main([str(x) for x in args], plugins=plugins)
        if len(rec) == 1:
            reprec = rec.pop()
        else:

            class reprec:
                pass
        reprec.ret = ret
        if ret == ExitCode.INTERRUPTED and (not no_reraise_ctrlc):
            calls = reprec.getcalls('pytest_keyboard_interrupt')
            if calls and calls[-1].excinfo.type == KeyboardInterrupt:
                raise KeyboardInterrupt()
        return reprec
    finally:
        for finalizer in finalizers:
            finalizer()