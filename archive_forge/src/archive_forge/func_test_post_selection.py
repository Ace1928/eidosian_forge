import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
@pytest.mark.parametrize('sim', [cirq.Simulator(), cirq.DensityMatrixSimulator()])
def test_post_selection(sim):
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    c = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q) ** 0.2, cirq.measure(q, key=key)), use_repetition_ids=False, repeat_until=cirq.KeyCondition(key)))
    result = sim.run(c)
    assert result.records['m'][0][-1] == (1,)
    for i in range(len(result.records['m'][0]) - 1):
        assert result.records['m'][0][i] == (0,)