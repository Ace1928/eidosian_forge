import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def run_ufunc(self, pyfunc, arg_types, arg_values):
    for tyargs, args in zip(arg_types, arg_values):
        cfunc = njit(tyargs)(pyfunc)
        got = cfunc(*args)
        expected = pyfunc(*_as_dtype_value(tyargs, args))
        msg = 'for args {0} typed {1}'.format(args, tyargs)
        special = set([(types.int32, types.uint64), (types.uint64, types.int32), (types.int64, types.uint64), (types.uint64, types.int64)])
        if tyargs in special:
            expected = float(expected)
        elif np.issubdtype(expected.dtype, np.inexact):
            expected = float(expected)
        elif np.issubdtype(expected.dtype, np.integer):
            expected = int(expected)
        elif np.issubdtype(expected.dtype, np.bool_):
            expected = bool(expected)
        alltypes = tyargs + (cfunc.overloads[tyargs].signature.return_type,)
        if any([t == types.float32 for t in alltypes]):
            prec = 'single'
        elif any([t == types.float64 for t in alltypes]):
            prec = 'double'
        else:
            prec = 'exact'
        self.assertPreciseEqual(got, expected, msg=msg, prec=prec)