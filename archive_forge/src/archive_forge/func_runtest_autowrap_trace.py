import sympy
import tempfile
import os
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip
def runtest_autowrap_trace(language, backend):
    has_module('numpy')
    trace = autowrap(A[i, i], language, backend)
    assert trace(numpy.eye(100)) == 100