import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_commutes_notimplemented_type():
    with pytest.raises(TypeError):
        cirq.commutes(cirq.SingleQubitCliffordGate.X, 'X')
    assert cirq.commutes(cirq.SingleQubitCliffordGate.X, 'X', default='default') == 'default'
    with pytest.raises(TypeError):
        cirq.commutes(cirq.CliffordGate.X, 'X')
    assert cirq.commutes(cirq.CliffordGate.X, 'X', default='default') == 'default'