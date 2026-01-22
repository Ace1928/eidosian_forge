from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def compute_pri_tol(self, eps_abs, eps_rel):
    """
        Compute primal tolerance using problem data
        """
    A = self.work.data.A
    if self.work.settings.scaling and (not self.work.settings.scaled_termination):
        Einv = self.work.scaling.Einv
        max_rel_eps = np.max([la.norm(Einv.dot(A.dot(self.work.x)), np.inf), la.norm(Einv.dot(self.work.z), np.inf)])
    else:
        max_rel_eps = np.max([la.norm(A.dot(self.work.x), np.inf), la.norm(self.work.z, np.inf)])
    eps_pri = eps_abs + eps_rel * max_rel_eps
    return eps_pri