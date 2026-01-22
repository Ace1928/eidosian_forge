import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('property_name', ('all_single_qubit_cliffords', 'CNOT', 'CZ', 'SWAP'))
def test_common_clifford_gate_caching(property_name):
    cache_name = f'_{property_name}'
    delattr(cirq.CliffordGate, cache_name)
    assert not hasattr(cirq.CliffordGate, cache_name)
    _ = getattr(cirq.CliffordGate, property_name)
    assert hasattr(cirq.CliffordGate, cache_name)