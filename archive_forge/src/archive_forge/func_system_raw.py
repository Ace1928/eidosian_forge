import abc
import ast
import atexit
import bdb
import builtins as builtin_mod
import functools
import inspect
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
import warnings
from ast import stmt
from io import open as io_open
from logging import error
from pathlib import Path
from typing import Callable
from typing import List as ListType, Dict as DictType, Any as AnyType
from typing import Optional, Sequence, Tuple
from warnings import warn
from tempfile import TemporaryDirectory
from traitlets import (
from traitlets.config.configurable import SingletonConfigurable
from traitlets.utils.importstring import import_item
import IPython.core.hooks
from IPython.core import magic, oinspect, page, prefilter, ultratb
from IPython.core.alias import Alias, AliasManager
from IPython.core.autocall import ExitAutocall
from IPython.core.builtin_trap import BuiltinTrap
from IPython.core.compilerop import CachingCompiler
from IPython.core.debugger import InterruptiblePdb
from IPython.core.display_trap import DisplayTrap
from IPython.core.displayhook import DisplayHook
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import InputRejected, UsageError
from IPython.core.events import EventManager, available_events
from IPython.core.extensions import ExtensionManager
from IPython.core.formatters import DisplayFormatter
from IPython.core.history import HistoryManager
from IPython.core.inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from IPython.core.logger import Logger
from IPython.core.macro import Macro
from IPython.core.payload import PayloadManager
from IPython.core.prefilter import PrefilterManager
from IPython.core.profiledir import ProfileDir
from IPython.core.usage import default_banner
from IPython.display import display
from IPython.paths import get_ipython_dir
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import PyColorize, io, openpy, py3compat
from IPython.utils.decorators import undoc
from IPython.utils.io import ask_yes_no
from IPython.utils.ipstruct import Struct
from IPython.utils.path import ensure_dir_exists, get_home_dir, get_py_filename
from IPython.utils.process import getoutput, system
from IPython.utils.strdispatch import StrDispatch
from IPython.utils.syspathcontext import prepended_to_syspath
from IPython.utils.text import DollarFormatter, LSString, SList, format_screen
from IPython.core.oinspect import OInfo
from ast import Module
from .async_helpers import (
def system_raw(self, cmd):
    """Call the given cmd in a subprocess using os.system on Windows or
        subprocess.call using the system shell on other platforms.

        Parameters
        ----------
        cmd : str
            Command to execute.
        """
    cmd = self.var_expand(cmd, depth=1)
    if cmd == '':
        main_cmd = ''
    else:
        main_cmd = cmd.split()[0]
    has_magic_alternatives = ('pip', 'conda', 'cd')
    if main_cmd in has_magic_alternatives:
        warnings.warn('You executed the system command !{0} which may not work as expected. Try the IPython magic %{0} instead.'.format(main_cmd))
    if sys.platform == 'win32':
        from IPython.utils._process_win32 import AvoidUNCPath
        with AvoidUNCPath() as path:
            if path is not None:
                cmd = '"pushd %s &&"%s' % (path, cmd)
            try:
                ec = os.system(cmd)
            except KeyboardInterrupt:
                print('\n' + self.get_exception_only(), file=sys.stderr)
                ec = -2
    else:
        executable = os.environ.get('SHELL', None)
        try:
            ec = subprocess.call(cmd, shell=True, executable=executable)
        except KeyboardInterrupt:
            print('\n' + self.get_exception_only(), file=sys.stderr)
            ec = 130
        if ec > 128:
            ec = -(ec - 128)
    self.user_ns['_exit_code'] = ec