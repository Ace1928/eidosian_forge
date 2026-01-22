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
def runtest_autowrap_matrix_vector(language, backend):
    has_module('numpy')
    x, y = symbols('x y', cls=IndexedBase)
    expr = Eq(y[i], A[i, j] * x[j])
    mv = autowrap(expr, language, backend)
    M = numpy.random.rand(10, 20)
    x = numpy.random.rand(20)
    y = numpy.dot(M, x)
    assert numpy.sum(numpy.abs(y - mv(M, x))) < 1e-13