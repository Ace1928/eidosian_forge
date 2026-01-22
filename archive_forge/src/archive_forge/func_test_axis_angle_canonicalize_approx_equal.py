import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_axis_angle_canonicalize_approx_equal():
    a1 = cirq.AxisAngleDecomposition(angle=np.pi, axis=(1, 0, 0), global_phase=1)
    a2 = cirq.AxisAngleDecomposition(angle=-np.pi, axis=(1, 0, 0), global_phase=-1)
    b1 = cirq.AxisAngleDecomposition(angle=np.pi, axis=(1, 0, 0), global_phase=-1)
    assert cirq.approx_eq(a1, a2, atol=1e-08)
    assert not cirq.approx_eq(a1, b1, atol=1e-08)