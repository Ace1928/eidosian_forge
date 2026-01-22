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
def test_balance_stoichiometry__duplicates():
    cases = '\nC + CO + CO2 -> C + CO        # suggested solution:  C +      CO2 ->     2 CO\nC + CO + CO2 -> C +      CO2  # suggested solution:      2 CO      -> C +      CO2\nC + CO + CO2 ->     CO + CO2  # suggested solution:  C +      CO2 ->     2 CO\nC + CO       -> C + CO + CO2  # suggested solution:      2 CO      -> C +      CO2\nC +      CO2 -> C + CO + CO2  # suggested solution:  C +      CO2 ->     2 CO\n    CO + CO2 -> C + CO + CO2  # suggested solution:      2 CO      -> C +      CO2\n'
    for prob, sol in [line.split('#') for line in cases.strip().splitlines()]:
        tst_r = Reaction.from_string(prob)
        ref_r = Reaction.from_string(sol.split(':')[1])
        tst_bal = balance_stoichiometry(tst_r.reac, tst_r.prod, allow_duplicates=True, underdetermined=None)
        assert Reaction(*tst_bal) == ref_r
    with pytest.raises(ValueError):
        balance_stoichiometry({'C', 'CO', 'CO2'}, {'C', 'CO', 'CO2'}, allow_duplicates=True, underdetermined=None)
    gh120 = ({'H4P2O7', 'HPO3', 'H2O'}, {'H4P2O7', 'HPO3'})
    bal120 = balance_stoichiometry(*gh120, allow_duplicates=True, underdetermined=None)
    assert bal120 == ({'HPO3': 2, 'H2O': 1}, {'H4P2O7': 1})
    with pytest.raises(ValueError):
        balance_stoichiometry(*gh120)
    bal_Mn = balance_stoichiometry({'H2O2', 'Mn1', 'H1'}, {'Mn1', 'H2O1'}, allow_duplicates=True, underdetermined=None)
    assert bal_Mn == ({'H2O2': 1, 'H1': 2}, {'H2O1': 2})
    bal_Mn_COx = balance_stoichiometry({'C', 'CO', 'CO2', 'Mn'}, {'C', 'CO2', 'Mn'}, allow_duplicates=True, underdetermined=None)
    assert bal_Mn_COx == ({'CO': 2}, {'C': 1, 'CO2': 1})