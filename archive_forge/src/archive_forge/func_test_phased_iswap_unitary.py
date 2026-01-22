import numpy as np
import pytest
import scipy
import sympy
import cirq
def test_phased_iswap_unitary():
    p = 0.3
    t = 0.4
    actual = cirq.unitary(cirq.PhasedISwapPowGate(phase_exponent=p, exponent=t))
    c = np.cos(np.pi * t / 2)
    s = np.sin(np.pi * t / 2) * 1j
    f = np.exp(2j * np.pi * p)
    expected = np.array([[1, 0, 0, 0], [0, c, s * f, 0], [0, s * f.conjugate(), c, 0], [0, 0, 0, 1]])
    assert np.allclose(actual, expected)