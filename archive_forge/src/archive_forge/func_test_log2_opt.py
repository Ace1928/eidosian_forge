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
def test_log2_opt():
    x = Symbol('x')
    expr1 = 7 * log(3 * x + 5) / log(2)
    opt1 = optimize(expr1, [log2_opt])
    assert opt1 == 7 * log2(3 * x + 5)
    assert opt1.rewrite(log) == expr1
    expr2 = 3 * log(5 * x + 7) / (13 * log(2))
    opt2 = optimize(expr2, [log2_opt])
    assert opt2 == 3 * log2(5 * x + 7) / 13
    assert opt2.rewrite(log) == expr2
    expr3 = log(x) / log(2)
    opt3 = optimize(expr3, [log2_opt])
    assert opt3 == log2(x)
    assert opt3.rewrite(log) == expr3
    expr4 = log(x) / log(2) + log(x + 1)
    opt4 = optimize(expr4, [log2_opt])
    assert opt4 == log2(x) + log(2) * log2(x + 1)
    assert opt4.rewrite(log) == expr4
    expr5 = log(17)
    opt5 = optimize(expr5, [log2_opt])
    assert opt5 == expr5
    expr6 = log(x + 3) / log(2)
    opt6 = optimize(expr6, [log2_opt])
    assert str(opt6) == 'log2(x + 3)'
    assert opt6.rewrite(log) == expr6