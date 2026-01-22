from subprocess import check_output
import os.path
from collections import defaultdict
import inspect
from functools import partial
import numba
from numba.core.registry import cpu_target
from all overloads.
def format_fname(fn):
    try:
        fname = '{0}.{1}'.format(fn.__module__, get_func_name(fn))
    except AttributeError:
        fname = repr(fn)
    return (fn, fname)