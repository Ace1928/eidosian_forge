import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
@contextmanager
def errorhandler(exty=None, usecase=None):
    try:
        yield
    except Exception as e:
        if exty is not None:
            lexty = exty if hasattr(exty, '__iter__') else [exty]
            found = False
            for ex in lexty:
                found |= isinstance(e, ex)
            if not found:
                raise
        else:
            should_not_fail.append((usecase, '%s: %s' % (type(e), str(e))))
    else:
        if exty is not None:
            should_fail.append(usecase)