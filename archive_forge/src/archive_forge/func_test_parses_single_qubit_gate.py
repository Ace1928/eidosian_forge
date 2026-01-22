import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_parses_single_qubit_gate(gate):
    assert gate == cirq.read_json(json_text=cirq.to_json(gate))