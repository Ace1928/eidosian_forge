import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_repr():
    cirq.testing.assert_equivalent_repr(cirq.KakDecomposition(global_phase=1j, single_qubit_operations_before=(cirq.unitary(cirq.X), cirq.unitary(cirq.Y)), interaction_coefficients=(0.3, 0.2, 0.1), single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z))))
    assert repr(cirq.KakDecomposition(global_phase=1, single_qubit_operations_before=(cirq.unitary(cirq.X), cirq.unitary(cirq.Y)), interaction_coefficients=(0.5, 0.25, 0), single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)))) == "\ncirq.KakDecomposition(\n    interaction_coefficients=(0.5, 0.25, 0),\n    single_qubit_operations_before=(\n        np.array([[0j, (1+0j)], [(1+0j), 0j]], dtype=np.dtype('complex128')),\n        np.array([[0j, -1j], [1j, 0j]], dtype=np.dtype('complex128')),\n    ),\n    single_qubit_operations_after=(\n        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.dtype('float64')),\n        np.array([[(1+0j), 0j], [0j, (-1+0j)]], dtype=np.dtype('complex128')),\n    ),\n    global_phase=1)\n".strip()