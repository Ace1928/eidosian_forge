import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
@pytest.mark.parametrize('add_measurements', [True, False])
@pytest.mark.parametrize('use_repetition_ids', [True, False])
@pytest.mark.parametrize('initial_reps', [0, 1, 2, 3])
def test_repeat_zero_times(add_measurements, use_repetition_ids, initial_reps):
    q = cirq.LineQubit(0)
    subcircuit = cirq.Circuit(cirq.X(q))
    if add_measurements:
        subcircuit.append(cirq.measure(q))
    op = cirq.CircuitOperation(subcircuit.freeze(), repetitions=initial_reps, use_repetition_ids=use_repetition_ids)
    result = cirq.Simulator().simulate(cirq.Circuit(op))
    assert np.allclose(result.state_vector(), [0, 1] if initial_reps % 2 else [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op ** 0))
    assert np.allclose(result.state_vector(), [1, 0])