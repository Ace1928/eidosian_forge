import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
@pytest.mark.xfail(run=False, reason='Needs to be fixed in `_highs_wrapper`')
def test_milp_options(capsys):
    message = "Unrecognized options detected: {'ekki'}..."
    options = {'ekki': True}
    with pytest.warns(RuntimeWarning, match=message):
        milp(1, options=options)
    A, b, c, numbers, M = magic_square(3)
    options = {'disp': True, 'presolve': False, 'time_limit': 0.05}
    res = milp(c=c, constraints=(A, b, b), bounds=(0, 1), integrality=1, options=options)
    captured = capsys.readouterr()
    assert 'Presolve is switched off' in captured.out
    assert 'Time Limit Reached' in captured.out
    assert not res.success