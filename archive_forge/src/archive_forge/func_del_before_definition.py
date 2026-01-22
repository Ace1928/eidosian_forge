import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def del_before_definition(rec):
    """
    This test reveal a bug that there is a del on uninitialized variable
    """
    n = 5
    for i in range(n):
        rec.mark(str(i))
        n = 0
        for j in range(n):
            return 0
        else:
            if i < 2:
                continue
            elif i == 2:
                for j in range(i):
                    return i
                rec.mark('FAILED')
            rec.mark('FAILED')
        rec.mark('FAILED')
    rec.mark('OK')
    return -1