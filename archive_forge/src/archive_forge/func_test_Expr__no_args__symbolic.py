import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
@requires('sympy')
def test_Expr__no_args__symbolic():
    K1 = MyK(unique_keys=('H1', 'S1'))
    K2 = MyK(unique_keys=('H2', 'S2'))
    add = K1 + K2
    import sympy
    v = defaultkeydict(sympy.Symbol)
    res = add(v, backend=sympy)
    R = 8.3145
    expr1 = sympy.exp(-(v['H1'] - v['T'] * v['S1']) / R / v['T'])
    expr2 = sympy.exp(-(v['H2'] - v['T'] * v['S2']) / R / v['T'])
    ref = expr1 + expr2
    assert (res - ref).simplify() == 0