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
def test_implicit_str():
    PolyT = create_Poly('x')
    p = PolyT([1, 2, 3])
    expr1 = p / 'u'
    assert abs(expr1({'x': 3, 'u': 5}) - (1 + 2 * 3 + 3 * 9) / 5) < 1e-12
    expr2 = 'u' / p
    assert abs(expr2({'x': 3, 'u': 5}) - 5 / (1 + 2 * 3 + 3 * 9)) < 1e-12
    assert expr1.all_parameter_keys() == set(['x'])
    assert expr2.all_parameter_keys() == set(['x'])