import numpy as np
import pytest
import scipy
import sympy
import cirq
def test_givens_rotation_equivalent_circuit():
    angle_rads = 3 * np.pi / 7
    t = 2 * angle_rads / np.pi
    gate = cirq.givens(angle_rads)
    q0, q1 = cirq.LineQubit.range(2)
    equivalent_circuit = cirq.Circuit([cirq.T(q0), cirq.T(q1) ** (-1), cirq.ISWAP(q0, q1) ** t, cirq.T(q0) ** (-1), cirq.T(q1)])
    assert np.allclose(cirq.unitary(gate), cirq.unitary(equivalent_circuit))