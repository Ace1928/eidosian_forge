from sympy.core import (S, pi, oo, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.logic import ITE
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import MatrixSymbol, SparseMatrix, Matrix
from sympy.printing.rust import rust_code
def test_reserved_words():
    x, y = symbols('x if')
    expr = sin(y)
    assert rust_code(expr) == 'if_.sin()'
    assert rust_code(expr, dereference=[y]) == '(*if_).sin()'
    assert rust_code(expr, reserved_word_suffix='_unreserved') == 'if_unreserved.sin()'
    with raises(ValueError):
        rust_code(expr, error_on_reserved=True)