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
def enable_pylab(self, gui=None, import_all=True, welcome_message=False):
    """Activate pylab support at runtime.

        This turns on support for matplotlib, preloads into the interactive
        namespace all of numpy and pylab, and configures IPython to correctly
        interact with the GUI event loop.  The GUI backend to be used can be
        optionally selected with the optional ``gui`` argument.

        This method only adds preloading the namespace to InteractiveShell.enable_matplotlib.

        Parameters
        ----------
        gui : optional, string
            If given, dictates the choice of matplotlib GUI backend to use
            (should be one of IPython's supported backends, 'qt', 'osx', 'tk',
            'gtk', 'wx' or 'inline'), otherwise we use the default chosen by
            matplotlib (as dictated by the matplotlib build-time options plus the
            user's matplotlibrc configuration file).  Note that not all backends
            make sense in all contexts, for example a terminal ipython can't
            display figures inline.
        import_all : optional, bool, default: True
            Whether to do `from numpy import *` and `from pylab import *`
            in addition to module imports.
        welcome_message : deprecated
            This argument is ignored, no welcome message will be displayed.
        """
    from IPython.core.pylabtools import import_pylab
    gui, backend = self.enable_matplotlib(gui)
    ns = {}
    import_pylab(ns, import_all)
    ignored = {'__builtins__'}
    both = set(ns).intersection(self.user_ns).difference(ignored)
    clobbered = [name for name in both if self.user_ns[name] is not ns[name]]
    self.user_ns.update(ns)
    self.user_ns_hidden.update(ns)
    return (gui, backend, clobbered)