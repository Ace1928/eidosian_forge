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
def enable_matplotlib(self, gui=None):
    """Enable interactive matplotlib and inline figure support.

        This takes the following steps:

        1. select the appropriate eventloop and matplotlib backend
        2. set up matplotlib for interactive use with that backend
        3. configure formatters for inline figure display
        4. enable the selected gui eventloop

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
        """
    from IPython.core import pylabtools as pt
    gui, backend = pt.find_gui_and_backend(gui, self.pylab_gui_select)
    if gui != 'inline':
        if self.pylab_gui_select is None:
            self.pylab_gui_select = gui
        elif gui != self.pylab_gui_select:
            print('Warning: Cannot change to a different GUI toolkit: %s. Using %s instead.' % (gui, self.pylab_gui_select))
            gui, backend = pt.find_gui_and_backend(self.pylab_gui_select)
    pt.activate_matplotlib(backend)
    from matplotlib_inline.backend_inline import configure_inline_support
    configure_inline_support(self, backend)
    self.enable_gui(gui)
    self.magics_manager.registry['ExecutionMagics'].default_runner = pt.mpl_runner(self.safe_execfile)
    return (gui, backend)