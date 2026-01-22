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
def test_3_1_disp_simplicial(self):
    """Iterative sampling on TestFunction 1 and 2  (multi and univariate)
        """

    def callback_func(x):
        print('Local minimization callback test')
    for test in [test1_1, test2_1]:
        shgo(test.f, test.bounds, iters=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})
        shgo(test.f, test.bounds, n=1, sampling_method='simplicial', callback=callback_func, options={'disp': True})