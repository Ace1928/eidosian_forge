from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
def test_issue_23807():
    eq = x ** 2 + y ** 2 + z ** 2 - 1000000
    base_soln = {(0, 0, 1000), (0, 352, 936), (480, 600, 640), (24, 640, 768), (192, 640, 744), (192, 480, 856), (168, 224, 960), (0, 600, 800), (280, 576, 768), (152, 480, 864), (0, 280, 960), (352, 360, 864), (424, 480, 768), (360, 480, 800), (224, 600, 768), (96, 360, 928), (168, 576, 800), (96, 480, 872)}
    assert diophantine(eq) == base_soln