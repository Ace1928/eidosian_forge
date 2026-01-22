import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def cm_cond(cond, inner_cm):
    with cond:
        with inner_cm as value:
            yield value