import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def is_neg_int(builder, val):
    return builder.icmp_signed('<', val, val.type(0))