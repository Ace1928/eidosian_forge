from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Field_preprocess():
    assert Field.preprocess(False) is False
    assert Field.preprocess(True) is True
    assert Field.preprocess(0) is False
    assert Field.preprocess(1) is True
    raises(OptionError, lambda: Field.preprocess(x))