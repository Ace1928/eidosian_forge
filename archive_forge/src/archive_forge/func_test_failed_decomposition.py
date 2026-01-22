import cirq
import pytest
import numpy as np
def test_failed_decomposition():
    with pytest.raises(ValueError):
        cirq.testing.assert_unitary_is_consistent(FailsOnDecompostion())
    _ = cirq.testing.assert_unitary_is_consistent(cirq.Circuit())