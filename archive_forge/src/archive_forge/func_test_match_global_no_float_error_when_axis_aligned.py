import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_match_global_no_float_error_when_axis_aligned():
    a = np.array([[1, 1.1], [-1.3, np.pi]])
    a2, _ = cirq.match_global_phase(a, a)
    a3, _ = cirq.match_global_phase(a * 1j, a * 1j)
    a4, _ = cirq.match_global_phase(-a, -a)
    a5, _ = cirq.match_global_phase(a * -1j, a * -1j)
    assert np.all(a2 == a)
    assert np.all(a3 == a)
    assert np.all(a4 == a)
    assert np.all(a5 == a)