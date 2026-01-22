from sympy.assumptions.ask import (Q, ask)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.core.symbol import symbols
from sympy.matrices.expressions.fourier import DFT, IDFT
from sympy.matrices import det, Matrix, Identity
from sympy.testing.pytest import raises
def test_dft2():
    assert DFT(1).as_explicit() == Matrix([[1]])
    assert DFT(2).as_explicit() == 1 / sqrt(2) * Matrix([[1, 1], [1, -1]])
    assert DFT(4).as_explicit() == Matrix([[S.Half, S.Half, S.Half, S.Half], [S.Half, -I / 2, Rational(-1, 2), I / 2], [S.Half, Rational(-1, 2), S.Half, Rational(-1, 2)], [S.Half, I / 2, Rational(-1, 2), -I / 2]])