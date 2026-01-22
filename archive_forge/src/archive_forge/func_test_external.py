import numpy as np
import pytest
import cirq
def test_external():
    for t in ['a', 1j]:
        cirq.testing.assert_equivalent_repr(t)
        cirq.testing.assert_equivalent_repr(t, setup_code='')
    cirq.testing.assert_equivalent_repr(np.array([5]), setup_code='from numpy import array')
    with pytest.raises(AssertionError, match='not defined'):
        cirq.testing.assert_equivalent_repr(np.array([5]))