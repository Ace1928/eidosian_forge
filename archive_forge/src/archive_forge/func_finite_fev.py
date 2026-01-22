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
def finite_fev(self):
    if self.disp:
        logging.info(f'Function evaluations done = {self.fn} / {self.maxfev}')
    if self.fn >= self.maxfev:
        self.stop_global = True
    return self.stop_global