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
def run_cell_magic(self, magic_name, line, cell):
    """Execute the given cell magic.

        Parameters
        ----------
        magic_name : str
            Name of the desired magic function, without '%' prefix.
        line : str
            The rest of the first input line as a single string.
        cell : str
            The body of the cell as a (possibly multiline) string.
        """
    fn = self._find_with_lazy_load('cell', magic_name)
    if fn is None:
        lm = self.find_line_magic(magic_name)
        etpl = 'Cell magic `%%{0}` not found{1}.'
        extra = '' if lm is None else ' (But line magic `%{0}` exists, did you mean that instead?)'.format(magic_name)
        raise UsageError(etpl.format(magic_name, extra))
    elif cell == '':
        message = '%%{0} is a cell magic, but the cell body is empty.'.format(magic_name)
        if self.find_line_magic(magic_name) is not None:
            message += ' Did you mean the line magic %{0} (single %)?'.format(magic_name)
        raise UsageError(message)
    else:
        stack_depth = 2
        if getattr(fn, magic.MAGIC_NO_VAR_EXPAND_ATTR, False):
            magic_arg_s = line
        else:
            magic_arg_s = self.var_expand(line, stack_depth)
        kwargs = {}
        if getattr(fn, 'needs_local_scope', False):
            kwargs['local_ns'] = self.user_ns
        with self.builtin_trap:
            args = (magic_arg_s, cell)
            result = fn(*args, **kwargs)
        if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):
            if DisplayHook.semicolon_at_end_of_expression(cell):
                return None
        return result