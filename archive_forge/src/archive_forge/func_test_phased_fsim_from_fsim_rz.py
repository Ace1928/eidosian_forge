import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('theta, phi, rz_angles_before, rz_angles_after', ((0, 0, (0, 0), (0, 0)), (1, 2, (3, 4), (5, 7)), (np.pi / 5, np.pi / 6, (0.1, 0.2), (0.3, 0.5))))
def test_phased_fsim_from_fsim_rz(theta, phi, rz_angles_before, rz_angles_after):
    f = cirq.PhasedFSimGate.from_fsim_rz(theta, phi, rz_angles_before, rz_angles_after)
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.rz(rz_angles_before[0]).on(q0), cirq.rz(rz_angles_before[1]).on(q1), cirq.FSimGate(theta, phi).on(q0, q1), cirq.rz(rz_angles_after[0]).on(q0), cirq.rz(rz_angles_after[1]).on(q1))
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(f), cirq.unitary(c), atol=1e-08)