import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_axis_angle_decomposition_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.AxisAngleDecomposition(angle=1, axis=(0.8, 0.6, 0), global_phase=-1))
    eq.add_equality_group(cirq.AxisAngleDecomposition(angle=5, axis=(0.8, 0.6, 0), global_phase=-1))
    eq.add_equality_group(cirq.AxisAngleDecomposition(angle=1, axis=(0.8, 0, 0.6), global_phase=-1))
    eq.add_equality_group(cirq.AxisAngleDecomposition(angle=1, axis=(0.8, 0.6, 0), global_phase=1))