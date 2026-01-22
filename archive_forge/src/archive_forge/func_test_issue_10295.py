from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, pi, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, Or, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.parsing.sympy_parser import null
from sympy.polys.polytools import Poly
from sympy.printing.repr import srepr
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.abc import x, y
from sympy.core.sympify import (sympify, _sympify, SympifyError, kernS,
from sympy.core.decorators import _sympifyit
from sympy.external import import_module
from sympy.testing.pytest import raises, XFAIL, skip, warns_deprecated_sympy
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.geometry import Point, Line
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.abc import _clash, _clash1, _clash2
from sympy.external.gmpy import HAS_GMPY
from sympy.sets import FiniteSet, EmptySet
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
import mpmath
from collections import defaultdict, OrderedDict
from mpmath.rational import mpq
def test_issue_10295():
    if not numpy:
        skip('numpy not installed.')
    A = numpy.array([[1, 3, -1], [0, 1, 7]])
    sA = S(A)
    assert sA.shape == (2, 3)
    for (ri, ci), val in numpy.ndenumerate(A):
        assert sA[ri, ci] == val
    B = numpy.array([-7, x, 3 * y ** 2])
    sB = S(B)
    assert sB.shape == (3,)
    assert B[0] == sB[0] == -7
    assert B[1] == sB[1] == x
    assert B[2] == sB[2] == 3 * y ** 2
    C = numpy.arange(0, 24)
    C.resize(2, 3, 4)
    sC = S(C)
    assert sC[0, 0, 0].is_integer
    assert sC[0, 0, 0] == 0
    a1 = numpy.array([1, 2, 3])
    a2 = numpy.array(list(range(24)))
    a2.resize(2, 4, 3)
    assert sympify(a1) == ImmutableDenseNDimArray([1, 2, 3])
    assert sympify(a2) == ImmutableDenseNDimArray(list(range(24)), (2, 4, 3))