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
def test_17_custom_sampling(self):
    """Test the functionality to add custom sampling methods to shgo"""

    def sample(n, d):
        return numpy.random.uniform(size=(n, d))
    run_test(test1_1, n=30, sampling_method=sample)