import pytest
import numpy as np
import cirq
def test_assert_consistent_mixture_not_normalized():
    mixture = _MixtureGate(0.1, 0.85)
    with pytest.raises(AssertionError, match='sum to 1'):
        cirq.testing.assert_consistent_mixture(mixture)
    mixture = _MixtureGate(0.2, 0.85)
    with pytest.raises(AssertionError, match='sum to 1'):
        cirq.testing.assert_consistent_mixture(mixture)