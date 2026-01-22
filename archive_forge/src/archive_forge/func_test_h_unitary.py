import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_h_unitary():
    sqrt = cirq.unitary(cirq.H ** 0.5)
    m = np.dot(sqrt, sqrt)
    assert np.allclose(m, cirq.unitary(cirq.H), atol=1e-08)