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
def test_7_2_minkwargs(self):
    """Test the minimizer_kwargs default inits"""
    minimizer_kwargs = {'ftol': 1e-05}
    options = {'disp': True}
    SHGO(test3_1.f, test3_1.bounds, constraints=test3_1.cons[0], minimizer_kwargs=minimizer_kwargs, options=options)