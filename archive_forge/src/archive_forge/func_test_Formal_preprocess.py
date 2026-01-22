from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Formal_preprocess():
    assert Formal.preprocess(False) is False
    assert Formal.preprocess(True) is True
    assert Formal.preprocess(0) is False
    assert Formal.preprocess(1) is True
    raises(OptionError, lambda: Formal.preprocess(x))