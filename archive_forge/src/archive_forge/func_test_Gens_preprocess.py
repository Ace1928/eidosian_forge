from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Gens_preprocess():
    assert Gens.preprocess((None,)) == ()
    assert Gens.preprocess((x, y, z)) == (x, y, z)
    assert Gens.preprocess(((x, y, z),)) == (x, y, z)
    a = Symbol('a', commutative=False)
    raises(GeneratorsError, lambda: Gens.preprocess((x, x, y)))
    raises(GeneratorsError, lambda: Gens.preprocess((x, y, a)))