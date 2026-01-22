from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Greedy_preprocess():
    assert Greedy.preprocess(False) is False
    assert Greedy.preprocess(True) is True
    assert Greedy.preprocess(0) is False
    assert Greedy.preprocess(1) is True
    raises(OptionError, lambda: Greedy.preprocess(x))