import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_to_clifford_tableau_util_function():
    tableau = cirq.ops.clifford_gate._to_clifford_tableau(x_to=(cirq.X, False), z_to=(cirq.Z, False))
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=0)
    tableau = cirq.ops.clifford_gate._to_clifford_tableau(x_to=(cirq.X, False), z_to=(cirq.Z, True))
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=1)
    tableau = cirq.ops.clifford_gate._to_clifford_tableau(rotation_map={cirq.X: (cirq.X, False), cirq.Z: (cirq.Z, False)})
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=0)
    tableau = cirq.ops.clifford_gate._to_clifford_tableau(rotation_map={cirq.X: (cirq.X, False), cirq.Z: (cirq.Z, True)})
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=1)
    with pytest.raises(ValueError):
        cirq.ops.clifford_gate._to_clifford_tableau()