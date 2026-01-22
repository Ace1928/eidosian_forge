import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (BFGS, SR1)
Rosenbrock function.

    The following optimization problem:
        minimize sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    