import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('theta, phi', [(0, 0), (np.pi / 3, np.pi / 5), (-np.pi / 3, np.pi / 5), (np.pi / 3, -np.pi / 5), (-np.pi / 3, -np.pi / 5), (np.pi / 2, 0.5)])
def test_fsim_consistent(theta, phi):
    gate = cirq.FSimGate(theta=theta, phi=phi)
    cirq.testing.assert_implements_consistent_protocols(gate)