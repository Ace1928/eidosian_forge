from __future__ import print_function
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import (atleast_2d_column_default,
from patsy.desc import Term, INTERCEPT
from patsy.build import *
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo
def test_same_factor_in_two_matrices():
    data = {'x': [1, 2, 3], 'a': ['a1', 'a2', 'a1']}

    def iter_maker():
        yield data
    t1 = make_termlist(['x'])
    t2 = make_termlist(['x', 'a'])
    builders = design_matrix_builders([t1, t2], iter_maker, eval_env=0)
    m1, m2 = build_design_matrices(builders, data)
    check_design_matrix(m1, 1, t1, column_names=['x'])
    assert np.allclose(m1, [[1], [2], [3]])
    check_design_matrix(m2, 2, t2, column_names=['x:a[a1]', 'x:a[a2]'])
    assert np.allclose(m2, [[1, 0], [0, 2], [3, 0]])