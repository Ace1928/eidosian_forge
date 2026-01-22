from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def is_dual_infeasible(self, eps_dual_inf):
    """
        Check dual infeasibility
            ||P*v||_inf = 0
        with v = delta_x / ||delta_x||_inf given that the following
        conditions hold
            q'* v < 0 and
                        | 0     if l_i, u_i \\in R
            (A * v)_i = { >= 0  if u_i = +inf
                        | <= 0  if l_i = -inf
        """
    if self.work.settings.scaling and (not self.work.settings.scaled_termination):
        norm_delta_x = la.norm(self.work.scaling.D.dot(self.work.delta_x), np.inf)
        scale_cost = self.work.scaling.c
    else:
        norm_delta_x = la.norm(self.work.delta_x, np.inf)
        scale_cost = 1.0
    if norm_delta_x > eps_dual_inf:
        if self.work.data.q.dot(self.work.delta_x) < -scale_cost * eps_dual_inf * norm_delta_x:
            self.work.Pdelta_x = self.work.data.P.dot(self.work.delta_x)
            if self.work.settings.scaling and (not self.work.settings.scaled_termination):
                self.work.Pdelta_x = self.work.scaling.Dinv.dot(self.work.Pdelta_x)
            if la.norm(self.work.Pdelta_x, np.inf) < scale_cost * eps_dual_inf * norm_delta_x:
                self.work.Adelta_x = self.work.data.A.dot(self.work.delta_x)
                if self.work.settings.scaling and (not self.work.settings.scaled_termination):
                    self.work.Adelta_x = self.work.scaling.Einv.dot(self.work.Adelta_x)
                for i in range(self.work.data.m):
                    if self.work.data.u[i] < OSQP_INFTY * MIN_SCALING and self.work.Adelta_x[i] > eps_dual_inf * norm_delta_x or (self.work.data.l[i] > -OSQP_INFTY * MIN_SCALING and self.work.Adelta_x[i] < -eps_dual_inf * norm_delta_x):
                        return False
                return True
    return False