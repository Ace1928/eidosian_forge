import random
import concurrent.futures
from collections.abc import Hashable
from sympy.core.add import Add
from sympy.core.function import (Function, diff, expand)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.printing.str import sstr
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.matrices.matrices import (ShapeError, MatrixError,
from sympy.matrices import (
from sympy.matrices.utilities import _dotprodsimp_state
from sympy.core import Tuple, Wild
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.utilities.iterables import flatten, capture, iterable
from sympy.utilities.exceptions import ignore_warnings, SymPyDeprecationWarning
from sympy.testing.pytest import (raises, XFAIL, slow, skip, skip_under_pyodide,
from sympy.assumptions import Q
from sympy.tensor.array import Array
from sympy.matrices.expressions import MatPow
from sympy.algebras import Quaternion
from sympy.abc import a, b, c, d, x, y, z, t
def test_Matrix_berkowitz_charpoly():
    UA, K_i, K_w = symbols('UA K_i K_w')
    A = Matrix([[-K_i - UA + K_i ** 2 / (K_i + K_w), K_i * K_w / (K_i + K_w)], [K_i * K_w / (K_i + K_w), -K_w + K_w ** 2 / (K_i + K_w)]])
    charpoly = A.charpoly(x)
    assert charpoly == Poly(x ** 2 + (K_i * UA + K_w * UA + 2 * K_i * K_w) / (K_i + K_w) * x + K_i * K_w * UA / (K_i + K_w), x, domain='ZZ(K_i,K_w,UA)')
    assert type(charpoly) is PurePoly
    A = Matrix([[1, 3], [2, 0]])
    assert A.charpoly() == A.charpoly(x) == PurePoly(x ** 2 - x - 6)
    A = Matrix([[1, 2], [x, 0]])
    p = A.charpoly(x)
    assert p.gen != x
    assert p.as_expr().subs(p.gen, x) == x ** 2 - 3 * x