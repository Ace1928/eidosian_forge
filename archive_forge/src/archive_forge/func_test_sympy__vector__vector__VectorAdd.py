import os
import re
from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import SKIP
from sympy.stats.crv_types import NormalDistribution
from sympy.stats.frv_types import DieDistribution
from sympy.matrices.expressions import MatrixSymbol
def test_sympy__vector__vector__VectorAdd():
    from sympy.vector.vector import VectorAdd, VectorMul
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    from sympy.abc import a, b, c, x, y, z
    v1 = a * C.i + b * C.j + c * C.k
    v2 = x * C.i + y * C.j + z * C.k
    assert _test_args(VectorAdd(v1, v2))
    assert _test_args(VectorMul(x, v1))