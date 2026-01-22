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
def finite_precision(self):
    """
        Stop the algorithm if the final function value is known

        Specify in options (with ``self.f_min_true = options['f_min']``)
        and the tolerance with ``f_tol = options['f_tol']``
        """
    self.find_lowest_vertex()
    if self.disp:
        logging.info(f'Lowest function evaluation = {self.f_lowest}')
        logging.info(f'Specified minimum = {self.f_min_true}')
    if self.f_lowest is None:
        return self.stop_global
    if self.f_min_true == 0.0:
        if self.f_lowest <= self.f_tol:
            self.stop_global = True
    else:
        pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
        if self.f_lowest <= self.f_min_true:
            self.stop_global = True
            if abs(pe) >= 2 * self.f_tol:
                warnings.warn(f'A much lower value than expected f* = {self.f_min_true} was found f_lowest = {self.f_lowest}', stacklevel=3)
        if pe <= self.f_tol:
            self.stop_global = True
    return self.stop_global