import numpy as np
import scipy.stats
import cirq
def test_decompose_size_unitary():
    np.random.seed(0)
    u = _random_unitary()
    assert _decomposition_size(u, 0) == (1, 0, 0)
    assert _decomposition_size(u, 1) == (4, 2, 0)
    assert _decomposition_size(u, 2) == (12, 8, 0)
    assert _decomposition_size(u, 3) == (20, 12, 2)
    assert _decomposition_size(u, 4) == (44, 28, 6)
    assert _decomposition_size(u, 5) == (84, 56, 18)
    assert _decomposition_size(u, 6) == (172, 120, 26)
    assert _decomposition_size(u, 7) == (340, 244, 38)
    assert _decomposition_size(u, 8) == (524, 380, 46)
    assert _decomposition_size(u, 9) == (820, 600, 58)