from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_fallback_two_qubit_random():
    for _ in range(25):
        state = cirq.testing.random_superposition(4)
        rho = np.outer(np.conjugate(state), state)
        u = cirq.testing.random_unitary(4)
        expected = 0.5 * rho + 0.5 * np.dot(np.dot(u, rho), np.conjugate(np.transpose(u)))
        rho.shape = (2, 2, 2, 2)
        expected.shape = (2, 2, 2, 2)

        class HasMixture:

            def _mixture_(self):
                return ((0.5, np.eye(4, dtype=np.complex128)), (0.5, u))
        assert_apply_mixture_returns(HasMixture(), rho, [0, 1], [2, 3], assert_result_is_out_buf=True, expected_result=expected)