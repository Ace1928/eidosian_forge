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
def test_4_2_bound_err(self):
    """Specified bounds are of the form (lb, ub)"""
    bounds = [(3, 5, 5), (3, 5)]
    assert_raises(ValueError, shgo, test1_1.f, bounds)