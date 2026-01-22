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
def sort_cache_result(self):
    """
        Sort results and build the global return object
        """
    results = {}
    self.xl_maps = np.array(self.xl_maps)
    self.f_maps = np.array(self.f_maps)
    ind_sorted = np.argsort(self.f_maps)
    results['xl'] = self.xl_maps[ind_sorted]
    self.f_maps = np.array(self.f_maps)
    results['funl'] = self.f_maps[ind_sorted]
    results['funl'] = results['funl'].T
    results['x'] = self.xl_maps[ind_sorted[0]]
    results['fun'] = self.f_maps[ind_sorted[0]]
    self.xl_maps = np.ndarray.tolist(self.xl_maps)
    self.f_maps = np.ndarray.tolist(self.f_maps)
    return results