import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_cnot_unitary():
    np.testing.assert_almost_equal(cirq.unitary(cirq.CNOT ** 0.5), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5 + 0.5j, 0.5 - 0.5j], [0, 0, 0.5 - 0.5j, 0.5 + 0.5j]]))