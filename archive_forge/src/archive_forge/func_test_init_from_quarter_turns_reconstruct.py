import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_init_from_quarter_turns_reconstruct(gate):
    new_gate = functools.reduce(cirq.SingleQubitCliffordGate.merged_with, (cirq.SingleQubitCliffordGate.from_quarter_turns(pauli, qt) for pauli, qt in gate.decompose_rotation()), cirq.SingleQubitCliffordGate.I)
    assert gate == new_gate