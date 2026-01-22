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
def test_19_parallelization(self):
    """Test the functionality to add custom sampling methods to shgo"""
    with Pool(2) as p:
        run_test(test1_1, n=30, workers=p.map)
    run_test(test1_1, n=30, workers=map)
    with Pool(2) as p:
        run_test(test_s, n=30, workers=p.map)
    run_test(test_s, n=30, workers=map)