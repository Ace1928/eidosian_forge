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
def runtest_issue_15337(language, backend):
    has_module('numpy')
    a, b, c, d, e = symbols('a, b, c, d, e')
    expr = (a - b + c - d + e) ** 13
    exp_res = (1.0 - 2.0 + 3.0 - 4.0 + 5.0) ** 13
    f = autowrap(expr, language, backend, args=(a, b, c, d, e), helpers=('f1', a - b + c, (a, b, c)))
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)
    f = autowrap(expr, language, backend, args=(a, b, c, d, e), helpers=(('f1', a - b, (a, b)), ('f2', c - d, (c, d))))
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)