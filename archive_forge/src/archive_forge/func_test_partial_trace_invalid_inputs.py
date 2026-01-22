import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace_invalid_inputs():
    with pytest.raises(ValueError, match='2, 3, 2, 2'):
        cirq.partial_trace(np.reshape(np.arange(2 * 3 * 2 * 2), (2, 3, 2, 2)), [1])
    with pytest.raises(ValueError, match='2'):
        cirq.partial_trace(np.reshape(np.arange(2 * 2 * 2 * 2), (2,) * 4), [2])