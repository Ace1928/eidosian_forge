import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def ifnot(builder, pred):
    return builder.if_then(builder.not_(pred))