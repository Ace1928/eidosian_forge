import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('theta, phi', ((0, 0), (0.1, 0.1), (-0.1, 0.1), (0.1, -0.1), (-0.1, -0.1), (np.pi / 2, np.pi / 6), (np.pi, np.pi), (3.5 * np.pi, 4 * np.pi)))
def test_fsim_iswap_cphase(theta, phi):
    q0, q1 = (cirq.NamedQubit('q0'), cirq.NamedQubit('q1'))
    iswap = cirq.ISWAP ** (-theta * 2 / np.pi)
    cphase = cirq.CZPowGate(exponent=-phi / np.pi)
    iswap_cphase = cirq.Circuit((iswap.on(q0, q1), cphase.on(q0, q1)))
    fsim = cirq.FSimGate(theta=theta, phi=phi)
    assert np.allclose(cirq.unitary(iswap_cphase), cirq.unitary(fsim))