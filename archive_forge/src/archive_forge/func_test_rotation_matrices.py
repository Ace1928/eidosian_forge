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
def test_rotation_matrices():
    theta = pi / 3
    r3_plus = rot_axis3(theta)
    r3_minus = rot_axis3(-theta)
    r2_plus = rot_axis2(theta)
    r2_minus = rot_axis2(-theta)
    r1_plus = rot_axis1(theta)
    r1_minus = rot_axis1(-theta)
    assert r3_minus * r3_plus * eye(3) == eye(3)
    assert r2_minus * r2_plus * eye(3) == eye(3)
    assert r1_minus * r1_plus * eye(3) == eye(3)
    assert r1_plus.trace() == 1 + 2 * cos(theta)
    assert r2_plus.trace() == 1 + 2 * cos(theta)
    assert r3_plus.trace() == 1 + 2 * cos(theta)
    assert rot_axis1(0) == eye(3)
    assert rot_axis2(0) == eye(3)
    assert rot_axis3(0) == eye(3)
    q1 = Quaternion.from_axis_angle([1, 0, 0], pi / 2)
    q2 = Quaternion.from_axis_angle([0, 1, 0], pi / 2)
    q3 = Quaternion.from_axis_angle([0, 0, 1], pi / 2)
    assert rot_axis1(-pi / 2) == q1.to_rotation_matrix()
    assert rot_axis2(-pi / 2) == q2.to_rotation_matrix()
    assert rot_axis3(-pi / 2) == q3.to_rotation_matrix()
    assert rot_ccw_axis1(+pi / 2) == q1.to_rotation_matrix()
    assert rot_ccw_axis2(+pi / 2) == q2.to_rotation_matrix()
    assert rot_ccw_axis3(+pi / 2) == q3.to_rotation_matrix()