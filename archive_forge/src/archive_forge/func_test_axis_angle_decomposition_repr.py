import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_axis_angle_decomposition_repr():
    cirq.testing.assert_equivalent_repr(cirq.AxisAngleDecomposition(angle=1, axis=(0, 0.6, 0.8), global_phase=-1))