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
@magic_arguments.magic_arguments()
@magic_arguments.argument('name', type=str, default='default', nargs='?')
@magic_arguments.argument('--remove', action='store_true', help='remove the current transformer')
@magic_arguments.argument('--list', action='store_true', help='list existing transformers name')
@magic_arguments.argument('--list-all', action='store_true', help='list existing transformers name and code template')
@line_cell_magic
def code_wrap(self, line, cell=None):
    """
        Simple magic to quickly define a code transformer for all IPython's future imput.

        ``__code__`` and ``__ret__`` are special variable that represent the code to run
        and the value of the last expression of ``__code__`` respectively.

        Examples
        --------

        .. ipython::

            In [1]: %%code_wrap before_after
               ...: print('before')
               ...: __code__
               ...: print('after')
               ...: __ret__


            In [2]: 1
            before
            after
            Out[2]: 1

            In [3]: %code_wrap --list
            before_after

            In [4]: %code_wrap --list-all
            before_after :
                print('before')
                __code__
                print('after')
                __ret__

            In [5]: %code_wrap --remove before_after

        """
    args = magic_arguments.parse_argstring(self.code_wrap, line)
    if args.list:
        for name in self._transformers.keys():
            print(name)
        return
    if args.list_all:
        for name, _t in self._transformers.items():
            print(name, ':')
            print(indent(ast.unparse(_t.template), '    '))
        print()
        return
    to_remove = self._transformers.pop(args.name, None)
    if to_remove in self.shell.ast_transformers:
        self.shell.ast_transformers.remove(to_remove)
    if cell is None or args.remove:
        return
    _trs = ReplaceCodeTransformer(ast.parse(cell))
    self._transformers[args.name] = _trs
    self.shell.ast_transformers.append(_trs)