import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_expectation_from_state_vector_basis_states():
    q = cirq.LineQubit.range(2)
    psum = cirq.X(q[0]) + 2 * cirq.Y(q[0]) + 3 * cirq.Z(q[0])
    q_map = {x: i for i, x in enumerate(q)}
    np.testing.assert_allclose(psum.expectation_from_state_vector(np.array([1, 1], dtype=complex) / np.sqrt(2), qubit_map=q_map), 1)
    np.testing.assert_allclose(psum.expectation_from_state_vector(np.array([1, -1], dtype=complex) / np.sqrt(2), qubit_map=q_map), -1)
    np.testing.assert_allclose(psum.expectation_from_state_vector(np.array([1, 1j], dtype=complex) / np.sqrt(2), qubit_map=q_map), 2)
    np.testing.assert_allclose(psum.expectation_from_state_vector(np.array([1, -1j], dtype=complex) / np.sqrt(2), qubit_map=q_map), -2)
    np.testing.assert_allclose(psum.expectation_from_state_vector(np.array([1, 0], dtype=complex), qubit_map=q_map), 3)
    np.testing.assert_allclose(psum.expectation_from_state_vector(np.array([0, 1], dtype=complex), qubit_map=q_map), -3)