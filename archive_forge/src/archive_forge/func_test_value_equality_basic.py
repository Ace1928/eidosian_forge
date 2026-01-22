import pytest
import cirq
def test_value_equality_basic():
    v = {BasicC(1): 4, BasicCa(2): 5}
    assert v[BasicCa(1)] == v[BasicC(1)] == 4
    assert v[BasicCa(2)] == 5
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(BasicC(1), BasicC(1), BasicCa(1), BasicCb(1))
    eq.add_equality_group(BasicD(1))
    eq.add_equality_group(BasicC(2))
    eq.add_equality_group(BasicCa(3))