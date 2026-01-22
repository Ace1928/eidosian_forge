from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
def test_Reaction__copy():
    r1 = Reaction({'H2O'}, {'H2O'}, checks=())
    r2 = r1.copy()
    assert r1 == r2
    r2.reac['H2O2'] = r2.reac.pop('H2O')
    r2.prod['H2O2'] = r2.prod.pop('H2O')
    assert r1.reac == {'H2O': 1} and r1.prod == {'H2O': 1}