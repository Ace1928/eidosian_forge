import numpy as np
import pytest
import cirq
def test_apply_channel_channel_fallback_two_qubit_random():
    for _ in range(25):
        state = cirq.testing.random_superposition(4)
        rho = np.outer(np.conjugate(state), state)
        u = cirq.testing.random_unitary(4)
        expected = 0.5 * rho + 0.5 * np.dot(np.dot(u, rho), np.conjugate(np.transpose(u)))
        rho.shape = (2, 2, 2, 2)
        expected.shape = (2, 2, 2, 2)

        class HasChannel:

            def _kraus_(self):
                return (np.sqrt(0.5) * np.eye(4, dtype=np.complex128), np.sqrt(0.5) * u)
        result = apply_channel(HasChannel(), rho, [0, 1], [2, 3], assert_result_is_out_buf=True)
        np.testing.assert_almost_equal(result, expected)