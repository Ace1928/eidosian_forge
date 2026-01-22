import numpy as np
import cirq
def test_via_has_stabilizer_effect_method():
    assert not cirq.has_stabilizer_effect(No1())
    assert not cirq.has_stabilizer_effect(No2())
    assert not cirq.has_stabilizer_effect(No3())
    assert cirq.has_stabilizer_effect(Yes())