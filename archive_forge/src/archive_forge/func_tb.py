import ast
import bdb
import builtins as builtin_mod
import copy
import cProfile as profile
import gc
import itertools
import math
import os
import pstats
import re
import shlex
import sys
import time
import timeit
from typing import Dict, Any
from ast import (
from io import StringIO
from logging import error
from pathlib import Path
from pdb import Restart
from textwrap import dedent, indent
from warnings import warn
from IPython.core import magic_arguments, oinspect, page
from IPython.core.displayhook import DisplayHook
from IPython.core.error import UsageError
from IPython.core.macro import Macro
from IPython.core.magic import (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.capture import capture_output
from IPython.utils.contexts import preserve_keys
from IPython.utils.ipstruct import Struct
from IPython.utils.module_paths import find_mod
from IPython.utils.path import get_py_filename, shellglob
from IPython.utils.timing import clock, clock2
from IPython.core.magics.ast_mod import ReplaceCodeTransformer
@line_magic
def tb(self, s):
    """Print the last traceback.

        Optionally, specify an exception reporting mode, tuning the
        verbosity of the traceback. By default the currently-active exception
        mode is used. See %xmode for changing exception reporting modes.

        Valid modes: Plain, Context, Verbose, and Minimal.
        """
    interactive_tb = self.shell.InteractiveTB
    if s:

        def xmode_switch_err(name):
            warn('Error changing %s exception modes.\n%s' % (name, sys.exc_info()[1]))
        new_mode = s.strip().capitalize()
        original_mode = interactive_tb.mode
        try:
            try:
                interactive_tb.set_mode(mode=new_mode)
            except Exception:
                xmode_switch_err('user')
            else:
                self.shell.showtraceback()
        finally:
            interactive_tb.set_mode(mode=original_mode)
    else:
        self.shell.showtraceback()