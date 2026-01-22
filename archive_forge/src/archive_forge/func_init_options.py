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
def init_options(self, options):
    """
        Initiates the options.

        Can also be useful to change parameters after class initiation.

        Parameters
        ----------
        options : dict

        Returns
        -------
        None

        """
    self.minimizer_kwargs['options'].update(options)
    for opt in ['jac', 'hess', 'hessp']:
        if opt in self.minimizer_kwargs['options']:
            self.minimizer_kwargs[opt] = self.minimizer_kwargs['options'].pop(opt)
    self.minimize_every_iter = options.get('minimize_every_iter', True)
    self.maxiter = options.get('maxiter', None)
    self.maxfev = options.get('maxfev', None)
    self.maxev = options.get('maxev', None)
    self.init = time.time()
    self.maxtime = options.get('maxtime', None)
    if 'f_min' in options:
        self.f_min_true = options['f_min']
        self.f_tol = options.get('f_tol', 0.0001)
    else:
        self.f_min_true = None
    self.minhgrd = options.get('minhgrd', None)
    self.symmetry = options.get('symmetry', False)
    if self.symmetry:
        self.symmetry = [0] * len(self.bounds)
    else:
        self.symmetry = None
    self.local_iter = options.get('local_iter', False)
    self.infty_cons_sampl = options.get('infty_constraints', True)
    self.disp = options.get('disp', False)