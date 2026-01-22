from sympy.core import (S, pi, oo, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.logic import ITE
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import MatrixSymbol, SparseMatrix, Matrix
from sympy.printing.rust import rust_code
def test_Functions():
    assert rust_code(sin(x) ** cos(x)) == 'x.sin().powf(x.cos())'
    assert rust_code(abs(x)) == 'x.abs()'
    assert rust_code(ceiling(x)) == 'x.ceil()'
    assert rust_code(floor(x)) == 'x.floor()'
    assert rust_code(Mod(x, 3)) == 'x - 3*((1_f64/3.0)*x).floor()'