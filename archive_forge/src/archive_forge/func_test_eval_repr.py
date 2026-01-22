from typing import cast
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('key', ['q0_1_0', cirq.MeasurementKey(name='q0_1_0'), cirq.MeasurementKey(path=('a', 'b'), name='c')])
def test_eval_repr(key):
    op = cirq.GateOperation(gate=cirq.MeasurementGate(1, key), qubits=[cirq.GridQubit(0, 1)])
    cirq.testing.assert_equivalent_repr(op)