import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_ry_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(cirq.unitary(cirq.ry(np.pi / 2)), np.array([[s, -s], [s, s]]))
    np.testing.assert_allclose(cirq.unitary(cirq.ry(-np.pi / 2)), np.array([[s, s], [-s, s]]))
    np.testing.assert_allclose(cirq.unitary(cirq.ry(0)), np.array([[1, 0], [0, 1]]))
    np.testing.assert_allclose(cirq.unitary(cirq.ry(2 * np.pi)), np.array([[-1, 0], [0, -1]]))
    np.testing.assert_allclose(cirq.unitary(cirq.ry(np.pi)), np.array([[0, -1], [1, 0]]))
    np.testing.assert_allclose(cirq.unitary(cirq.ry(-np.pi)), np.array([[0, 1], [-1, 0]]))