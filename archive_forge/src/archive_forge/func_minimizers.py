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
def minimizers(self):
    """
        Returns the indexes of all minimizers
        """
    self.minimizer_pool = []
    for x in self.HC.V.cache:
        in_LMC = False
        if len(self.LMC.xl_maps) > 0:
            for xlmi in self.LMC.xl_maps:
                if np.all(np.array(x) == np.array(xlmi)):
                    in_LMC = True
        if in_LMC:
            continue
        if self.HC.V[x].minimiser():
            if self.disp:
                logging.info('=' * 60)
                logging.info(f'v.x = {self.HC.V[x].x_a} is minimizer')
                logging.info(f'v.f = {self.HC.V[x].f} is minimizer')
                logging.info('=' * 30)
            if self.HC.V[x] not in self.minimizer_pool:
                self.minimizer_pool.append(self.HC.V[x])
            if self.disp:
                logging.info('Neighbors:')
                logging.info('=' * 30)
                for vn in self.HC.V[x].nn:
                    logging.info(f'x = {vn.x} || f = {vn.f}')
                logging.info('=' * 60)
    self.minimizer_pool_F = []
    self.X_min = []
    self.X_min_cache = {}
    for v in self.minimizer_pool:
        self.X_min.append(v.x_a)
        self.minimizer_pool_F.append(v.f)
        self.X_min_cache[tuple(v.x_a)] = v.x
    self.minimizer_pool_F = np.array(self.minimizer_pool_F)
    self.X_min = np.array(self.X_min)
    self.sort_min_pool()
    return self.X_min