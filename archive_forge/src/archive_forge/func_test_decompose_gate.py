import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('clifford_gate', cirq.SingleQubitCliffordGate.all_single_qubit_cliffords)
def test_decompose_gate(clifford_gate):
    gates = clifford_gate.decompose_gate()
    u = functools.reduce(np.dot, [np.eye(2), *(cirq.unitary(gate) for gate in reversed(gates))])
    assert np.allclose(u, cirq.unitary(clifford_gate))