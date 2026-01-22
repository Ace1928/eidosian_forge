from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Modulus_preprocess():
    assert Modulus.preprocess(23) == 23
    assert Modulus.preprocess(Integer(23)) == 23
    raises(OptionError, lambda: Modulus.preprocess(0))
    raises(OptionError, lambda: Modulus.preprocess(x))