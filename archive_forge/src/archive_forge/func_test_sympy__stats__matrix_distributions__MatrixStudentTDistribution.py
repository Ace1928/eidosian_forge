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
def test_sympy__stats__matrix_distributions__MatrixStudentTDistribution():
    from sympy.stats.matrix_distributions import MatrixStudentTDistribution
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    v = symbols('v', positive=True)
    Omega = MatrixSymbol('Omega', 3, 3)
    Sigma = MatrixSymbol('Sigma', 1, 1)
    Location = MatrixSymbol('Location', 1, 3)
    assert _test_args(MatrixStudentTDistribution(v, Location, Omega, Sigma))