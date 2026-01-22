import os
import sys
import runpy
import types
from . import get_start_method, set_start_method
from . import process
from .context import reduction
from . import util
def get_command_line(**kwds):
    """
    Returns prefix of command line used for spawning a child process
    """
    if getattr(sys, 'frozen', False):
        return [sys.executable, '--multiprocessing-fork'] + ['%s=%r' % item for item in kwds.items()]
    else:
        prog = 'from multiprocess.spawn import spawn_main; spawn_main(%s)'
        prog %= ', '.join(('%s=%r' % item for item in kwds.items()))
        opts = util._args_from_interpreter_flags()
        return [_python_exe] + opts + ['-c', prog, '--multiprocessing-fork']