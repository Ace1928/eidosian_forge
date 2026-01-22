import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_clifford_gate_act_on_ch_form():
    args = cirq.StabilizerChFormSimulationState(initial_state=cirq.StabilizerStateChForm(num_qubits=2, initial_state=1), qubits=cirq.LineQubit.range(2), prng=np.random.RandomState())
    cirq.act_on(cirq.CliffordGate.X, args, qubits=cirq.LineQubit.range(1))
    np.testing.assert_allclose(args.state.state_vector(), np.array([0, 0, 0, 1]))