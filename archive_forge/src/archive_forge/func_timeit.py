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
@no_var_expand
@line_cell_magic
@needs_local_scope
def timeit(self, line='', cell=None, local_ns=None):
    """Time execution of a Python statement or expression

        Usage, in line mode:
          %timeit [-n<N> -r<R> [-t|-c] -q -p<P> -o] statement
        or in cell mode:
          %%timeit [-n<N> -r<R> [-t|-c] -q -p<P> -o] setup_code
          code
          code...

        Time execution of a Python statement or expression using the timeit
        module.  This function can be used both as a line and cell magic:

        - In line mode you can time a single-line statement (though multiple
          ones can be chained with using semicolons).

        - In cell mode, the statement in the first line is used as setup code
          (executed but not timed) and the body of the cell is timed.  The cell
          body has access to any variables created in the setup code.

        Options:
        -n<N>: execute the given statement <N> times in a loop. If <N> is not
        provided, <N> is determined so as to get sufficient accuracy.

        -r<R>: number of repeats <R>, each consisting of <N> loops, and take the
        average result.
        Default: 7

        -t: use time.time to measure the time, which is the default on Unix.
        This function measures wall time.

        -c: use time.clock to measure the time, which is the default on
        Windows and measures wall time. On Unix, resource.getrusage is used
        instead and returns the CPU user time.

        -p<P>: use a precision of <P> digits to display the timing result.
        Default: 3

        -q: Quiet, do not print result.

        -o: return a TimeitResult that can be stored in a variable to inspect
            the result in more details.

        .. versionchanged:: 7.3
            User variables are no longer expanded,
            the magic line is always left unmodified.

        Examples
        --------
        ::

          In [1]: %timeit pass
          8.26 ns ± 0.12 ns per loop (mean ± std. dev. of 7 runs, 100000000 loops each)

          In [2]: u = None

          In [3]: %timeit u is None
          29.9 ns ± 0.643 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

          In [4]: %timeit -r 4 u == None

          In [5]: import time

          In [6]: %timeit -n1 time.sleep(2)

        The times reported by %timeit will be slightly higher than those
        reported by the timeit.py script when variables are accessed. This is
        due to the fact that %timeit executes the statement in the namespace
        of the shell, compared with timeit.py, which uses a single setup
        statement to import function or create variables. Generally, the bias
        does not matter as long as results from timeit.py are not mixed with
        those from %timeit."""
    opts, stmt = self.parse_options(line, 'n:r:tcp:qo', posix=False, strict=False, preserve_non_opts=True)
    if stmt == '' and cell is None:
        return
    timefunc = timeit.default_timer
    number = int(getattr(opts, 'n', 0))
    default_repeat = 7 if timeit.default_repeat < 7 else timeit.default_repeat
    repeat = int(getattr(opts, 'r', default_repeat))
    precision = int(getattr(opts, 'p', 3))
    quiet = 'q' in opts
    return_result = 'o' in opts
    if hasattr(opts, 't'):
        timefunc = time.time
    if hasattr(opts, 'c'):
        timefunc = clock
    timer = Timer(timer=timefunc)
    transform = self.shell.transform_cell
    if cell is None:
        ast_setup = self.shell.compile.ast_parse('pass')
        ast_stmt = self.shell.compile.ast_parse(transform(stmt))
    else:
        ast_setup = self.shell.compile.ast_parse(transform(stmt))
        ast_stmt = self.shell.compile.ast_parse(transform(cell))
    ast_setup = self.shell.transform_ast(ast_setup)
    ast_stmt = self.shell.transform_ast(ast_stmt)
    self.shell.compile(ast_setup, '<magic-timeit-setup>', 'exec')
    self.shell.compile(ast_stmt, '<magic-timeit-stmt>', 'exec')
    timeit_ast_template = ast.parse('def inner(_it, _timer):\n    setup\n    _t0 = _timer()\n    for _i in _it:\n        stmt\n    _t1 = _timer()\n    return _t1 - _t0\n')
    timeit_ast = TimeitTemplateFiller(ast_setup, ast_stmt).visit(timeit_ast_template)
    timeit_ast = ast.fix_missing_locations(timeit_ast)
    tc_min = 0.1
    t0 = clock()
    code = self.shell.compile(timeit_ast, '<magic-timeit>', 'exec')
    tc = clock() - t0
    ns = {}
    glob = self.shell.user_ns
    conflict_globs = {}
    if local_ns and cell is None:
        for var_name, var_val in glob.items():
            if var_name in local_ns:
                conflict_globs[var_name] = var_val
        glob.update(local_ns)
    exec(code, glob, ns)
    timer.inner = ns['inner']
    if number == 0:
        for index in range(0, 10):
            number = 10 ** index
            time_number = timer.timeit(number)
            if time_number >= 0.2:
                break
    all_runs = timer.repeat(repeat, number)
    best = min(all_runs) / number
    worst = max(all_runs) / number
    timeit_result = TimeitResult(number, repeat, best, worst, all_runs, tc, precision)
    if conflict_globs:
        glob.update(conflict_globs)
    if not quiet:
        if worst > 4 * best and best > 0 and (worst > 1e-06):
            print('The slowest run took %0.2f times longer than the fastest. This could mean that an intermediate result is being cached.' % (worst / best))
        print(timeit_result)
        if tc > tc_min:
            print('Compiler time: %.2f s' % tc)
    if return_result:
        return timeit_result