import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_state_vector_2():
    q = cirq.LineQubit.range(2)
    rs = np.random.RandomState(52)
    for _ in range(10):
        g = cirq.MatrixGate(cirq.testing.random_unitary(dim=2 ** len(q), random_state=rs))
        c = cirq.Circuit(g.on(*q))
        psi1 = cirq.final_state_vector(c, dtype=np.complex128)
        psi2 = ccq.tensor_state_vector(c, q)
        np.testing.assert_allclose(psi1, psi2, atol=1e-08)