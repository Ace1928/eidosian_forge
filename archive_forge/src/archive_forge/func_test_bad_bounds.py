import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy.optimize._linprog_util import _clean_inputs, _LPProblem
from scipy._lib._util import VisibleDeprecationWarning
from copy import deepcopy
from datetime import date
def test_bad_bounds():
    lp = _LPProblem(c=[1, 2])
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=(1, 2, 2)))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2, 2)]))
    with np.testing.suppress_warnings() as sup:
        sup.filter(VisibleDeprecationWarning, 'Creating an ndarray from ragged')
        assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2), (1, 2, 2)]))
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2), (1, 2), (1, 2)]))
    lp = _LPProblem(c=[1, 2, 3, 4])
    assert_raises(ValueError, _clean_inputs, lp._replace(bounds=[(1, 2, 3, 4), (1, 2, 3, 4)]))