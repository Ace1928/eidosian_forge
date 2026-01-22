import re
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_implements_consistent_protocols(n):
    u = cirq.testing.random_unitary(2 ** n)
    g1 = cirq.MatrixGate(u)
    cirq.testing.assert_implements_consistent_protocols(g1, ignoring_global_phase=True)
    cirq.testing.assert_decompose_ends_at_default_gateset(g1)
    if n == 1:
        return
    g2 = cirq.MatrixGate(u, qid_shape=(4,) + (2,) * (n - 2))
    cirq.testing.assert_implements_consistent_protocols(g2, ignoring_global_phase=True)
    cirq.testing.assert_decompose_ends_at_default_gateset(g2)