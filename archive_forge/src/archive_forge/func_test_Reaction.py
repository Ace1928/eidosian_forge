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
def test_Reaction():
    substances = s_Hp, s_OHm, s_H2O = (Substance('H+', composition={0: 1, 1: 1}), Substance('OH-', composition={0: -1, 1: 1, 8: 1}), Substance('H2O', composition={0: 0, 1: 2, 8: 1}))
    substance_names = Hp, OHm, H2O = [s.name for s in substances]
    substance_dict = {n: s for n, s in zip(substance_names, substances)}
    r1 = Reaction({Hp: 1, OHm: 1}, {H2O: 1})
    assert sum(r1.composition_violation(substance_dict)) == 0
    assert r1.composition_violation(substance_dict, ['H+']) == [0]
    viol, cks = r1.composition_violation(substance_dict, True)
    assert viol == [0] * 3 and sorted(cks) == [0, 1, 8]
    assert r1.charge_neutrality_violation(substance_dict) == 0
    r2 = Reaction({Hp: 1, OHm: 1}, {H2O: 2})
    assert sum(r2.composition_violation(substance_dict)) != 0
    assert r2.charge_neutrality_violation(substance_dict) == 0
    r3 = Reaction({Hp: 2, OHm: 1}, {H2O: 2})
    assert sum(r3.composition_violation(substance_dict)) != 0
    assert r3.charge_neutrality_violation(substance_dict) != 0
    assert r3.keys() == {Hp, OHm, H2O}
    with pytest.raises(ValueError):
        Reaction({Hp: -1, OHm: -1}, {H2O: -1})
    assert r1 == Reaction({'H+', 'OH-'}, {'H2O'})
    r4 = Reaction({Hp, OHm}, {H2O}, 7)
    ref = {Hp: -3 * 5 * 7, OHm: -3 * 5 * 7, H2O: 3 * 5 * 7}
    r4.rate({Hp: 5, OHm: 3}) == ref
    r5 = r4.copy()
    assert r5 == r4
    assert r5 != r1
    lhs5, rhs5 = ({'H+': 1, 'OH-': 1}, {'H2O': 1})
    r5 = Reaction(lhs5, rhs5)
    assert r5.reac == lhs5 and r5.prod == rhs5