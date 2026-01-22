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
def test_lj_symmetry_old(self):
    """LJ: Symmetry-constrained test function"""
    options = {'symmetry': True, 'disp': True}
    args = (6,)
    run_test(testLJ, args=args, n=300, options=options, iters=1, sampling_method='simplicial')