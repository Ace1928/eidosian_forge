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
@requires('sympy')
def test_balance_stoichiometry__ordering():
    reac, prod = ('CuSCN KIO3 HCl'.split(), 'CuSO4 KCl HCN ICl H2O'.split())
    rxn = Reaction(*balance_stoichiometry(reac, prod))
    res = rxn.string()
    ref = '4 CuSCN + 7 KIO3 + 14 HCl -> 4 CuSO4 + 7 KCl + 4 HCN + 7 ICl + 5 H2O'
    assert res == ref