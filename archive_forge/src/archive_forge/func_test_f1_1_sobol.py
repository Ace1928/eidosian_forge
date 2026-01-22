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
def test_f1_1_sobol(self):
    """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(-1, 6), (-1, 6)]"""
    run_test(test1_1)