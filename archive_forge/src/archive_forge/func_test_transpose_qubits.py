from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
def test_transpose_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    args = ExampleSimulationState()
    assert args.transpose_to_qubit_order((q1, q0)).qubits == (q1, q0)
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q2))
    with pytest.raises(ValueError, match='Qubits do not match'):
        args.transpose_to_qubit_order((q0, q1, q1))