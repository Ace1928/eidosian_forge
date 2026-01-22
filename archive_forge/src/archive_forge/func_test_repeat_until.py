import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_repeat_until(sim):
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    c = cirq.Circuit(cirq.X(q), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q), cirq.measure(q, key=key)), use_repetition_ids=False, repeat_until=cirq.KeyCondition(key)))
    measurements = sim.run(c).records['m'][0]
    assert len(measurements) == 2
    assert measurements[0] == (0,)
    assert measurements[1] == (1,)