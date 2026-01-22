import numpy as np
import pytest
import cirq
def test_tp_projector():
    q0, q1 = cirq.LineQubit.range(2)
    p00 = (cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1)))
    np.testing.assert_allclose(rho, p00)
    p01 = (cirq.KET_ZERO(q0) * cirq.KET_ONE(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit([cirq.I.on_each(q0, q1), cirq.X(q1)]))
    np.testing.assert_allclose(rho, p01)
    ppp = (cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit([cirq.H.on_each(q0, q1)]))
    np.testing.assert_allclose(rho, ppp, atol=1e-07)
    ppm = (cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit([cirq.H.on_each(q0, q1), cirq.Z(q1)]))
    np.testing.assert_allclose(rho, ppm, atol=1e-07)
    pii = (cirq.KET_IMAG(q0) * cirq.KET_IMAG(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit(cirq.rx(-np.pi / 2).on_each(q0, q1)))
    np.testing.assert_allclose(rho, pii, atol=1e-07)
    pij = (cirq.KET_IMAG(q0) * cirq.KET_MINUS_IMAG(q1)).projector()
    rho = cirq.final_density_matrix(cirq.Circuit(cirq.rx(-np.pi / 2)(q0), cirq.rx(np.pi / 2)(q1)))
    np.testing.assert_allclose(rho, pij, atol=1e-07)