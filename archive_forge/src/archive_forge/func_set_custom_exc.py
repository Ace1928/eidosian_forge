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
def set_custom_exc(self, exc_tuple, handler):
    """set_custom_exc(exc_tuple, handler)

        Set a custom exception handler, which will be called if any of the
        exceptions in exc_tuple occur in the mainloop (specifically, in the
        run_code() method).

        Parameters
        ----------
        exc_tuple : tuple of exception classes
            A *tuple* of exception classes, for which to call the defined
            handler.  It is very important that you use a tuple, and NOT A
            LIST here, because of the way Python's except statement works.  If
            you only want to trap a single exception, use a singleton tuple::

                exc_tuple == (MyCustomException,)

        handler : callable
            handler must have the following signature::

                def my_handler(self, etype, value, tb, tb_offset=None):
                    ...
                    return structured_traceback

            Your handler must return a structured traceback (a list of strings),
            or None.

            This will be made into an instance method (via types.MethodType)
            of IPython itself, and it will be called if any of the exceptions
            listed in the exc_tuple are caught. If the handler is None, an
            internal basic one is used, which just prints basic info.

            To protect IPython from crashes, if your handler ever raises an
            exception or returns an invalid result, it will be immediately
            disabled.

        Notes
        -----
        WARNING: by putting in your own exception handler into IPython's main
        execution loop, you run a very good chance of nasty crashes.  This
        facility should only be used if you really know what you are doing.
        """
    if not isinstance(exc_tuple, tuple):
        raise TypeError('The custom exceptions must be given as a tuple.')

    def dummy_handler(self, etype, value, tb, tb_offset=None):
        print('*** Simple custom exception handler ***')
        print('Exception type :', etype)
        print('Exception value:', value)
        print('Traceback      :', tb)

    def validate_stb(stb):
        """validate structured traceback return type

            return type of CustomTB *should* be a list of strings, but allow
            single strings or None, which are harmless.

            This function will *always* return a list of strings,
            and will raise a TypeError if stb is inappropriate.
            """
        msg = 'CustomTB must return list of strings, not %r' % stb
        if stb is None:
            return []
        elif isinstance(stb, str):
            return [stb]
        elif not isinstance(stb, list):
            raise TypeError(msg)
        for line in stb:
            if not isinstance(line, str):
                raise TypeError(msg)
        return stb
    if handler is None:
        wrapped = dummy_handler
    else:

        def wrapped(self, etype, value, tb, tb_offset=None):
            """wrap CustomTB handler, to protect IPython from user code

                This makes it harder (but not impossible) for custom exception
                handlers to crash IPython.
                """
            try:
                stb = handler(self, etype, value, tb, tb_offset=tb_offset)
                return validate_stb(stb)
            except:
                self.set_custom_exc((), None)
                print('Custom TB Handler failed, unregistering', file=sys.stderr)
                stb = self.InteractiveTB.structured_traceback(*sys.exc_info())
                print(self.InteractiveTB.stb2text(stb))
                print('The original exception:')
                stb = self.InteractiveTB.structured_traceback((etype, value, tb), tb_offset=tb_offset)
            return stb
    self.CustomTB = types.MethodType(wrapped, self)
    self.custom_exceptions = exc_tuple