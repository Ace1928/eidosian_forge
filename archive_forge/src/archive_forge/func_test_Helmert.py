from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def test_Helmert():
    t1 = Helmert()
    for levels in (['a', 'b', 'c', 'd'], ('a', 'b', 'c', 'd')):
        matrix = t1.code_with_intercept(levels)
        assert matrix.column_suffixes == ['[H.intercept]', '[H.b]', '[H.c]', '[H.d]']
        assert np.allclose(matrix.matrix, [[1, -1, -1, -1], [1, 1, -1, -1], [1, 0, 2, -1], [1, 0, 0, 3]])
        matrix = t1.code_without_intercept(levels)
        assert matrix.column_suffixes == ['[H.b]', '[H.c]', '[H.d]']
        assert np.allclose(matrix.matrix, [[-1, -1, -1], [1, -1, -1], [0, 2, -1], [0, 0, 3]])