from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def reject_gen2(x):
    _ = object()
    a = np.arange(3)
    for i in range(a.size):
        res = a[i] + x
        for j in range(i):
            res = res ** 2
        yield res