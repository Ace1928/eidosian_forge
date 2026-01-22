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
def g2(x):
    return 12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3] - 21 - 1.645 * numpy.sqrt(0.28 * x[0] ** 2 + 0.19 * x[1] ** 2 + 20.5 * x[2] ** 2 + 0.62 * x[3] ** 2)