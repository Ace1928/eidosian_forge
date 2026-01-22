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
def test_Expr__no_args__arg_defaults():
    K1 = MyK2(unique_keys=('H1', 'S1', 'Cp1'))
    K2 = MyK2(unique_keys=('H2', 'S2'))
    add = K1 + K2
    assert add.all_unique_keys() == set(['H1', 'H2', 'S1', 'S2', 'Cp1'])
    T = 293.15
    res = add({'H1': 2, 'H2': 3, 'S1': 5, 'S2': 7, 'T': T, 'Cp1': 13})
    RT = 8.3145 * T
    H1p = 2 + 13 * (T - 298.15)
    S1p = 5 + 13 * math.log(T / 298.15)
    ref = math.exp(-(H1p - T * S1p) / RT) + math.exp(-(3 - T * 7) / RT)
    assert abs(res - ref) < 1e-14