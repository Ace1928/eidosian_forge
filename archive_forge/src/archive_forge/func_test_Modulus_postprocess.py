from sympy.polys.polyoptions import (
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z
def test_Modulus_postprocess():
    opt = {'modulus': 5}
    Modulus.postprocess(opt)
    assert opt == {'modulus': 5, 'domain': FF(5)}
    opt = {'modulus': 5, 'symmetric': False}
    Modulus.postprocess(opt)
    assert opt == {'modulus': 5, 'domain': FF(5, False), 'symmetric': False}