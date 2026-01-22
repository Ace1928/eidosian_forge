from collections import namedtuple
import time
import logging
import warnings
import sys
import numpy as np
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper
from scipy.optimize._shgo_lib._complex import Complex
def sampling_custom(self, n, dim):
    """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
    if self.n_sampled == 0:
        self.C = self.sampling_function(n, dim)
    else:
        self.C = self.sampling_function(n, dim)
    for i in range(len(self.bounds)):
        self.C[:, i] = self.C[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
    return self.C