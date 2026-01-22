import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('clifford_gate,standard_gate', [(cirq.CliffordGate.I, cirq.I), (cirq.CliffordGate.X, cirq.X), (cirq.CliffordGate.Y, cirq.Y), (cirq.CliffordGate.Z, cirq.Z), (cirq.CliffordGate.H, cirq.H), (cirq.CliffordGate.S, cirq.S), (cirq.CliffordGate.CNOT, cirq.CNOT), (cirq.CliffordGate.CZ, cirq.CZ), (cirq.CliffordGate.SWAP, cirq.SWAP)])
def test_common_clifford_gate(clifford_gate, standard_gate):
    u_c = cirq.unitary(clifford_gate)
    u_s = cirq.unitary(standard_gate)
    cirq.testing.assert_allclose_up_to_global_phase(u_c, u_s, atol=1e-08)