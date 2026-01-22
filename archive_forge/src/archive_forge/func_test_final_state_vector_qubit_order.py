import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_state_vector_qubit_order():
    a, b = cirq.LineQubit.range(2)
    np.testing.assert_allclose(cirq.final_state_vector([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[a, b]), [0, 0, 0.5 + 0.5j, 0.5 - 0.5j])
    np.testing.assert_allclose(cirq.final_state_vector([cirq.X(a), cirq.X(b) ** 0.5], qubit_order=[b, a]), [0, 0.5 + 0.5j, 0, 0.5 - 0.5j])