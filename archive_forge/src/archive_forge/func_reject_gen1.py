from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def reject_gen1(x):
    _ = object()
    a = np.arange(4)
    for i in range(a.shape[0]):
        yield a[i]