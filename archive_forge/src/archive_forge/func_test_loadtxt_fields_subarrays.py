import os
import numpy as np
from numpy.testing import (
def test_loadtxt_fields_subarrays(self):
    from io import StringIO
    dt = [('a', 'u1', 2), ('b', 'u1', 2)]
    x = np.loadtxt(StringIO('0 1 2 3'), dtype=dt)
    assert_equal(x, np.array([((0, 1), (2, 3))], dtype=dt))
    dt = [('a', [('a', 'u1', (1, 3)), ('b', 'u1')])]
    x = np.loadtxt(StringIO('0 1 2 3'), dtype=dt)
    assert_equal(x, np.array([(((0, 1, 2), 3),)], dtype=dt))
    dt = [('a', 'u1', (2, 2))]
    x = np.loadtxt(StringIO('0 1 2 3'), dtype=dt)
    assert_equal(x, np.array([(((0, 1), (2, 3)),)], dtype=dt))
    dt = [('a', 'u1', (2, 3, 2))]
    x = np.loadtxt(StringIO('0 1 2 3 4 5 6 7 8 9 10 11'), dtype=dt)
    data = [((((0, 1), (2, 3), (4, 5)), ((6, 7), (8, 9), (10, 11))),)]
    assert_equal(x, np.array(data, dtype=dt))