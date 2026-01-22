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
def test_issue_17247_expression_blowup_27():
    M = Matrix([[0, 1 - x, x + 1, 1 - x], [1 - x, x + 1, 0, x + 1], [0, 1 - x, x + 1, 1 - x], [0, 0, 1 - x, 0]])
    with dotprodsimp(True):
        P, J = M.jordan_form()
        assert P.expand() == Matrix(S('[\n            [    0,  4*x/(x**2 - 2*x + 1), -(-17*x**4 + 12*sqrt(2)*x**4 - 4*sqrt(2)*x**3 + 6*x**3 - 6*x - 4*sqrt(2)*x + 12*sqrt(2) + 17)/(-7*x**4 + 5*sqrt(2)*x**4 - 6*sqrt(2)*x**3 + 8*x**3 - 2*x**2 + 8*x + 6*sqrt(2)*x - 5*sqrt(2) - 7), -(12*sqrt(2)*x**4 + 17*x**4 - 6*x**3 - 4*sqrt(2)*x**3 - 4*sqrt(2)*x + 6*x - 17 + 12*sqrt(2))/(7*x**4 + 5*sqrt(2)*x**4 - 6*sqrt(2)*x**3 - 8*x**3 + 2*x**2 - 8*x + 6*sqrt(2)*x - 5*sqrt(2) + 7)],\n            [x - 1, x/(x - 1) + 1/(x - 1),                       (-7*x**3 + 5*sqrt(2)*x**3 - x**2 + sqrt(2)*x**2 - sqrt(2)*x - x - 5*sqrt(2) - 7)/(-3*x**3 + 2*sqrt(2)*x**3 - 2*sqrt(2)*x**2 + 3*x**2 + 2*sqrt(2)*x + 3*x - 3 - 2*sqrt(2)),                       (7*x**3 + 5*sqrt(2)*x**3 + x**2 + sqrt(2)*x**2 - sqrt(2)*x + x - 5*sqrt(2) + 7)/(2*sqrt(2)*x**3 + 3*x**3 - 3*x**2 - 2*sqrt(2)*x**2 - 3*x + 2*sqrt(2)*x - 2*sqrt(2) + 3)],\n            [    0,                     1,                                                                                            -(-3*x**2 + 2*sqrt(2)*x**2 + 2*x - 3 - 2*sqrt(2))/(-x**2 + sqrt(2)*x**2 - 2*sqrt(2)*x + 1 + sqrt(2)),                                                                                            -(2*sqrt(2)*x**2 + 3*x**2 - 2*x - 2*sqrt(2) + 3)/(x**2 + sqrt(2)*x**2 - 2*sqrt(2)*x - 1 + sqrt(2))],\n            [1 - x,                     0,                                                                                                                                                                                               1,                                                                                                                                                                                             1]]')).expand()
        assert J == Matrix(S('[\n            [0, 1,                       0,                       0],\n            [0, 0,                       0,                       0],\n            [0, 0, x - sqrt(2)*(x - 1) + 1,                       0],\n            [0, 0,                       0, x + sqrt(2)*(x - 1) + 1]]'))