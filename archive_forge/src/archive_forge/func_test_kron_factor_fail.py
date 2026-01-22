import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kron_factor_fail():
    mat = cirq.kron_with_controls(cirq.CONTROL_TAG, X)
    g, f1, f2 = cirq.kron_factor_4x4_to_2x2s(mat)
    with pytest.raises(ValueError):
        assert_kronecker_factorization_not_within_tolerance(mat, g, f1, f2)
    mat = cirq.kron_factor_4x4_to_2x2s(np.diag([1, 1, 1, 1j]))
    with pytest.raises(ValueError):
        assert_kronecker_factorization_not_within_tolerance(mat, g, f1, f2)