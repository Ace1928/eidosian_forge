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
def finite_homology_growth(self):
    """
        Stop the algorithm if homology group rank did not grow in iteration.
        """
    if self.LMC.size == 0:
        return
    self.hgrd = self.LMC.size - self.hgr
    self.hgr = self.LMC.size
    if self.hgrd <= self.minhgrd:
        self.stop_global = True
    if self.disp:
        logging.info(f'Current homology growth = {self.hgrd}  (minimum growth = {self.minhgrd})')
    return self.stop_global