import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_match_global_phase_incompatible_shape():
    a = np.array([1])
    b = np.array([1, 2])
    c, d = cirq.match_global_phase(a, b)
    assert c.shape == a.shape
    assert d.shape == b.shape
    assert c is not a
    assert d is not b
    assert np.allclose(c, a)
    assert np.allclose(d, b)
    a = np.array([])
    b = np.array([])
    c, d = cirq.match_global_phase(a, b)
    assert c.shape == a.shape
    assert d.shape == b.shape
    assert c is not a
    assert d is not b
    assert np.allclose(c, a)
    assert np.allclose(d, b)