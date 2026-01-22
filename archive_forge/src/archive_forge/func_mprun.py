from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
@line_cell_magic
def mprun(self, parameter_s='', cell=None):
    """ Execute a statement under the line-by-line memory profiler from the
        memory_profiler module.

        Usage, in line mode:
          %mprun -f func1 -f func2 <statement>

        Usage, in cell mode:
          %%mprun -f func1 -f func2 [statement]
          code...
          code...

        In cell mode, the additional code lines are appended to the (possibly
        empty) statement in the first line. Cell mode allows you to easily
        profile multiline blocks without having to put them in a separate
        function.

        The given statement (which doesn't require quote marks) is run via the
        LineProfiler. Profiling is enabled for the functions specified by the -f
        options. The statistics will be shown side-by-side with the code through
        the pager once the statement has completed.

        Options:

        -f <function>: LineProfiler only profiles functions and methods it is told
        to profile.  This option tells the profiler about these functions. Multiple
        -f options may be used. The argument may be any expression that gives
        a Python function or method object. However, one must be careful to avoid
        spaces that may confuse the option parser. Additionally, functions defined
        in the interpreter at the In[] prompt or via %run currently cannot be
        displayed.  Write these functions out to a separate file and import them.

        One or more -f options are required to get any useful results.

        -T <filename>: dump the text-formatted statistics with the code
        side-by-side out to a text file.

        -r: return the LineProfiler object after it has completed profiling.

        -c: If present, add the memory usage of any children process to the report.
        """
    from io import StringIO
    from memory_profiler import show_results, LineProfiler
    from distutils.version import LooseVersion
    import IPython
    ipython_version = LooseVersion(IPython.__version__)
    if ipython_version < '0.11':
        from IPython.genutils import page
        from IPython.ipstruct import Struct
        from IPython.ipapi import UsageError
    else:
        from IPython.core.page import page
        from IPython.utils.ipstruct import Struct
        from IPython.core.error import UsageError
    opts_def = Struct(T=[''], f=[])
    parameter_s = parameter_s.replace('"', '\\"').replace("'", "\\'")
    opts, arg_str = self.parse_options(parameter_s, 'rf:T:c', list_all=True)
    opts.merge(opts_def)
    global_ns = self.shell.user_global_ns
    local_ns = self.shell.user_ns
    if cell is not None:
        arg_str += '\n' + cell
    funcs = []
    for name in opts.f:
        try:
            funcs.append(eval(name, global_ns, local_ns))
        except Exception as e:
            raise UsageError('Could not find function %r.\n%s: %s' % (name, e.__class__.__name__, e))
    include_children = 'c' in opts
    profile = LineProfiler(include_children=include_children)
    for func in funcs:
        profile(func)
    if 'profile' in builtins.__dict__:
        had_profile = True
        old_profile = builtins.__dict__['profile']
    else:
        had_profile = False
        old_profile = None
    builtins.__dict__['profile'] = profile
    try:
        profile.runctx(arg_str, global_ns, local_ns)
        message = ''
    except SystemExit:
        message = '*** SystemExit exception caught in code being profiled.'
    except KeyboardInterrupt:
        message = '*** KeyboardInterrupt exception caught in code being profiled.'
    finally:
        if had_profile:
            builtins.__dict__['profile'] = old_profile
    stdout_trap = StringIO()
    show_results(profile, stdout_trap)
    output = stdout_trap.getvalue()
    output = output.rstrip()
    if ipython_version < '0.11':
        page(output, screen_lines=self.shell.rc.screen_length)
    else:
        page(output)
    print(message)
    text_file = opts.T[0]
    if text_file:
        with open(text_file, 'w') as pfile:
            pfile.write(output)
        print('\n*** Profile printout saved to text file %s. %s' % (text_file, message))
    return_value = None
    if 'r' in opts:
        return_value = profile
    return return_value