import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
def test_gh16971():

    def cons(x):
        return np.sum(x ** 2) - 0
    c = {'fun': cons, 'type': 'ineq'}
    minimizer_kwargs = {'method': 'COBYLA', 'options': {'rhobeg': 5, 'tol': 0.5, 'catol': 0.05}}
    s = SHGO(rosen, [(0, 10)] * 2, constraints=c, minimizer_kwargs=minimizer_kwargs)
    assert s.minimizer_kwargs['method'].lower() == 'cobyla'
    assert s.minimizer_kwargs['options']['catol'] == 0.05