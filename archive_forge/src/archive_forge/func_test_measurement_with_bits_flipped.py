from typing import cast
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('num_qubits, mask, bits, flipped', [(1, (), [0], (True,)), (3, (False,), [1], (False, True)), (3, (False, False), [0, 2], (True, False, True))])
def test_measurement_with_bits_flipped(num_qubits, mask, bits, flipped):
    gate = cirq.MeasurementGate(num_qubits, key='a', invert_mask=mask, qid_shape=(3,) * num_qubits)
    gate1 = gate.with_bits_flipped(*bits)
    assert gate1.key == gate.key
    assert gate1.num_qubits() == gate.num_qubits()
    assert gate1.invert_mask == flipped
    assert cirq.qid_shape(gate1) == cirq.qid_shape(gate)
    gate2 = gate1.with_bits_flipped(*bits)
    assert gate2.full_invert_mask() == gate.full_invert_mask()