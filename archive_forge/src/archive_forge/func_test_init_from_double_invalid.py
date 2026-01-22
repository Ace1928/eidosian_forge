import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('trans1,from1', ((trans1, from1) for trans1, from1 in itertools.product(_all_rotations(), _paulis)))
def test_init_from_double_invalid(trans1, from1):
    from2 = cirq.Pauli.by_relative_index(from1, 1)
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map({from1: trans1, from2: trans1})