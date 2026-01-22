from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def test_kraus_fallback_to_unitary():
    u = np.array([[1, 0], [1, 0]])

    class ReturnsUnitary:

        def _unitary_(self) -> np.ndarray:
            return u
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary()), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), None), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), NotImplemented), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), (1,)), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), LOCAL_DEFAULT), (u,))
    assert cirq.has_kraus(ReturnsUnitary())