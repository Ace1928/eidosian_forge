import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('trans_x,trans_z', _all_rotation_pairs())
def test_init_from_xz(trans_x, trans_z):
    gate = cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)
    assert gate.pauli_tuple(cirq.X) == trans_x
    assert gate.pauli_tuple(cirq.Z) == trans_z
    _assert_not_mirror(gate)
    _assert_no_collision(gate)