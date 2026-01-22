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
def test_8_homology_group_diff(self):
    options = {'minhgrd': 1, 'minimize_every_iter': True}
    run_test(test1_1, n=None, iters=None, options=options, sampling_method='simplicial')