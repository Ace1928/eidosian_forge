import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_match_global_phase():
    a = np.array([[5, 4], [3, -2]])
    b = np.array([[1e-06, 0], [0, 1j]])
    c, d = cirq.match_global_phase(a, b)
    np.testing.assert_allclose(c, -a, atol=1e-10)
    np.testing.assert_allclose(d, b * -1j, atol=1e-10)