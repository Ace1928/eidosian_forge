import numpy as np
import scipy.stats
import cirq
def test_decompose_random_unitary():
    for controls_count in range(5):
        for _ in range(10):
            _test_decompose(_random_unitary(), controls_count)
    for controls_count in range(5, 8):
        _test_decompose(_random_unitary(), controls_count)