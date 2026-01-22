from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet
from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin
from itertools import product
def test_issue_8247_8354():
    from sympy.functions.elementary.trigonometric import tan
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3 * sqrt(3)) - sqrt(10 + 6 * sqrt(3))
    assert z.is_positive is False
    z = S('-2**(1/3)*(3*sqrt(93) + 29)**2 - 4*(3*sqrt(93) + 29)**(4/3) +\n        12*sqrt(93)*(3*sqrt(93) + 29)**(1/3) + 116*(3*sqrt(93) + 29)**(1/3) +\n        174*2**(1/3)*sqrt(93) + 1678*2**(1/3)')
    assert z.is_positive is False
    z = 2 * (-3 * tan(19 * pi / 90) + sqrt(3)) * cos(11 * pi / 90) * cos(19 * pi / 90) - sqrt(3) * (-3 + 4 * cos(19 * pi / 90) ** 2)
    assert z.is_positive is not True
    z = S('9*(3*sqrt(93) + 29)**(2/3)*((3*sqrt(93) +\n        29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) - 2) - 2*2**(1/3))**3 +\n        72*(3*sqrt(93) + 29)**(2/3)*(81*sqrt(93) + 783) + (162*sqrt(93) +\n        1566)*((3*sqrt(93) + 29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) -\n        2) - 2*2**(1/3))**2')
    assert z.is_positive is False