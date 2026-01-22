import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_axis_angle_canonicalize():
    a = cirq.AxisAngleDecomposition(angle=np.pi * 2.3, axis=(1, 0, 0), global_phase=1j).canonicalize()
    assert a.global_phase == -1j
    assert a.axis == (1, 0, 0)
    np.testing.assert_allclose(a.angle, np.pi * 0.3, atol=1e-08)
    a = cirq.AxisAngleDecomposition(angle=np.pi / 2, axis=(-1, 0, 0), global_phase=1j).canonicalize()
    assert a.global_phase == 1j
    assert a.axis == (1, 0, 0)
    assert a.angle == -np.pi / 2
    a = cirq.AxisAngleDecomposition(angle=np.pi + 0.01, axis=(1, 0, 0), global_phase=1j).canonicalize(atol=0.1)
    assert a.global_phase == 1j
    assert a.axis == (1, 0, 0)
    assert a.angle == np.pi + 0.01
    a = cirq.AxisAngleDecomposition(angle=np.pi + 0.01, axis=(1, 0, 0), global_phase=1j).canonicalize(atol=0.001)
    assert a.global_phase == -1j
    assert a.axis == (1, 0, 0)
    assert np.isclose(a.angle, -np.pi + 0.01)