from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
def test_approximate_common_period():
    from cirq.ops.eigen_gate import _approximate_common_period as f
    assert f([]) is None
    assert f([0]) is None
    assert f([1, 0]) is None
    assert f([np.e, np.pi]) is None
    assert f([1]) == 1
    assert f([-1]) == 1
    assert f([2.5]) == 2.5
    assert f([1.5, 2]) == 6
    assert f([2, 3]) == 6
    assert abs(f([1 / 3, 2 / 3]) - 2 / 3) < 1e-08
    assert abs(f([2 / 5, 3 / 5]) - 6 / 5) < 1e-08
    assert f([0.5, -0.5]) == 0.5
    np.testing.assert_allclose(f([np.e]), np.e, atol=1e-08)