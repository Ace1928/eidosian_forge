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
def sampled_surface(self, infty_cons_sampl=False):
    """
        Sample the function surface.

        There are 2 modes, if ``infty_cons_sampl`` is True then the sampled
        points that are generated outside the feasible domain will be
        assigned an ``inf`` value in accordance with SHGO rules.
        This guarantees convergence and usually requires less objective
        function evaluations at the computational costs of more Delaunay
        triangulation points.

        If ``infty_cons_sampl`` is False, then the infeasible points are
        discarded and only a subspace of the sampled points are used. This
        comes at the cost of the loss of guaranteed convergence and usually
        requires more objective function evaluations.
        """
    if self.disp:
        logging.info('Generating sampling points')
    self.sampling(self.nc, self.dim)
    if len(self.LMC.xl_maps) > 0:
        self.C = np.vstack((self.C, np.array(self.LMC.xl_maps)))
    if not infty_cons_sampl:
        if self.g_cons is not None:
            self.sampling_subspace()
    self.sorted_samples()
    self.n_sampled = self.nc