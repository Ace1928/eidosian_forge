import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_exponent():
    cnot = cirq.PauliInteractionGate(cirq.Z, False, cirq.X, False)
    np.testing.assert_almost_equal(cirq.unitary(cnot ** 0.5), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5 + 0.5j, 0.5 - 0.5j], [0, 0, 0.5 - 0.5j, 0.5 + 0.5j]]))