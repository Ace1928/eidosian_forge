import tempfile
from sympy.core.numbers import pi, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions import assuming, Q
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.cfunctions import log2, exp2, expm1, log1p
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.codegen.rewriting import (
from sympy.testing.pytest import XFAIL, skip
from sympy.utilities import lambdify
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
def test_sinc_opts():

    def check(d):
        for k, v in d.items():
            assert optimize(k, sinc_opts) == v
    x = Symbol('x')
    check({sin(x) / x: sinc(x), sin(2 * x) / (2 * x): sinc(2 * x), sin(3 * x) / x: 3 * sinc(3 * x), x * sin(x): x * sin(x)})
    y = Symbol('y')
    check({sin(x * y) / (x * y): sinc(x * y), y * sin(x / y) / x: sinc(x / y), sin(sin(x)) / sin(x): sinc(sin(x)), sin(3 * sin(x)) / sin(x): 3 * sinc(3 * sin(x)), sin(x) / y: sin(x) / y})