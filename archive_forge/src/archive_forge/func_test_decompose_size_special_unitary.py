import numpy as np
import scipy.stats
import cirq
def test_decompose_size_special_unitary():
    np.random.seed(0)
    u = _random_special_unitary()
    assert _decomposition_size(u, 0) == (1, 0, 0)
    assert _decomposition_size(u, 1) == (3, 2, 0)
    assert _decomposition_size(u, 2) == (8, 8, 0)
    assert _decomposition_size(u, 3) == (8, 6, 2)
    assert _decomposition_size(u, 4) == (24, 18, 4)
    assert _decomposition_size(u, 5) == (40, 30, 12)
    for i in range(6, 20):
        assert _decomposition_size(u, i) == (64 * i - 312, 48 * i - 234, 16)