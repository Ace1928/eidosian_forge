import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_targeted_conjugate_multiple_indices():
    a = np.reshape(np.arange(16) + 1j, (2, 2, 2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [0, 1])
    expected = np.einsum('ijkl,klmn,mnop->ijop', a, b, np.transpose(np.conjugate(a), (2, 3, 0, 1)))
    np.testing.assert_almost_equal(result, expected)