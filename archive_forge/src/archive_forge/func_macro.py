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
@skip_doctest
@line_magic
def macro(self, parameter_s=''):
    """Define a macro for future re-execution. It accepts ranges of history,
        filenames or string objects.

        Usage:\\
          %macro [options] name n1-n2 n3-n4 ... n5 .. n6 ...

        Options:

          -r: use 'raw' input.  By default, the 'processed' history is used,
          so that magics are loaded in their transformed version to valid
          Python.  If this option is given, the raw input as typed at the
          command line is used instead.
          
          -q: quiet macro definition.  By default, a tag line is printed 
          to indicate the macro has been created, and then the contents of 
          the macro are printed.  If this option is given, then no printout
          is produced once the macro is created.

        This will define a global variable called `name` which is a string
        made of joining the slices and lines you specify (n1,n2,... numbers
        above) from your input history into a single string. This variable
        acts like an automatic function which re-executes those lines as if
        you had typed them. You just type 'name' at the prompt and the code
        executes.

        The syntax for indicating input ranges is described in %history.

        Note: as a 'hidden' feature, you can also use traditional python slice
        notation, where N:M means numbers N through M-1.

        For example, if your history contains (print using %hist -n )::

          44: x=1
          45: y=3
          46: z=x+y
          47: print x
          48: a=5
          49: print 'x',x,'y',y

        you can create a macro with lines 44 through 47 (included) and line 49
        called my_macro with::

          In [55]: %macro my_macro 44-47 49

        Now, typing `my_macro` (without quotes) will re-execute all this code
        in one pass.

        You don't need to give the line-numbers in order, and any given line
        number can appear multiple times. You can assemble macros with any
        lines from your input history in any order.

        The macro is a simple object which holds its value in an attribute,
        but IPython's display system checks for macros and executes them as
        code instead of printing them when you type their name.

        You can view a macro's contents by explicitly printing it with::

          print macro_name

        """
    opts, args = self.parse_options(parameter_s, 'rq', mode='list')
    if not args:
        return sorted((k for k, v in self.shell.user_ns.items() if isinstance(v, Macro)))
    if len(args) == 1:
        raise UsageError("%macro insufficient args; usage '%macro name n1-n2 n3-4...")
    name, codefrom = (args[0], ' '.join(args[1:]))
    try:
        lines = self.shell.find_user_code(codefrom, 'r' in opts)
    except (ValueError, TypeError) as e:
        print(e.args[0])
        return
    macro = Macro(lines)
    self.shell.define_macro(name, macro)
    if not 'q' in opts:
        print('Macro `%s` created. To execute, type its name (without quotes).' % name)
        print('=== Macro contents: ===')
        print(macro, end=' ')