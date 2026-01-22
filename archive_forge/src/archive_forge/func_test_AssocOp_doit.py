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
def test_AssocOp_doit():
    a = Add(x, x, evaluate=False)
    b = Mul(y, y, evaluate=False)
    c = Add(b, b, evaluate=False)
    d = Mul(a, a, evaluate=False)
    assert c.doit(deep=False).func == Mul
    assert c.doit(deep=False).args == (2, y, y)
    assert c.doit().func == Mul
    assert c.doit().args == (2, Pow(y, 2))
    assert d.doit(deep=False).func == Pow
    assert d.doit(deep=False).args == (a, 2 * S.One)
    assert d.doit().func == Mul
    assert d.doit().args == (4 * S.One, Pow(x, 2))