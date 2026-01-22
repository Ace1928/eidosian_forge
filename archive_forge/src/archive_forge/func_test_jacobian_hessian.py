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
def test_jacobian_hessian():
    L = Matrix(1, 2, [x ** 2 * y, 2 * y ** 2 + x * y])
    syms = [x, y]
    assert L.jacobian(syms) == Matrix([[2 * x * y, x ** 2], [y, 4 * y + x]])
    L = Matrix(1, 2, [x, x ** 2 * y ** 3])
    assert L.jacobian(syms) == Matrix([[1, 0], [2 * x * y ** 3, x ** 2 * 3 * y ** 2]])
    f = x ** 2 * y
    syms = [x, y]
    assert hessian(f, syms) == Matrix([[2 * y, 2 * x], [2 * x, 0]])
    f = x ** 2 * y ** 3
    assert hessian(f, syms) == Matrix([[2 * y ** 3, 6 * x * y ** 2], [6 * x * y ** 2, 6 * x ** 2 * y]])
    f = z + x * y ** 2
    g = x ** 2 + 2 * y ** 3
    ans = Matrix([[0, 2 * y], [2 * y, 2 * x]])
    assert ans == hessian(f, Matrix([x, y]))
    assert ans == hessian(f, Matrix([x, y]).T)
    assert hessian(f, (y, x), [g]) == Matrix([[0, 6 * y ** 2, 2 * x], [6 * y ** 2, 2 * x, 2 * y], [2 * x, 2 * y, 0]])