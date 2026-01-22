import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('angle_rads, expected_unitary', [(0, np.eye(4)), (1, np.diag([1, 1, 1, np.exp(1j)])), (np.pi / 2, np.diag([1, 1, 1, 1j]))])
def test_cphase_unitary(angle_rads, expected_unitary):
    np.testing.assert_allclose(cirq.unitary(cirq.cphase(angle_rads)), expected_unitary)