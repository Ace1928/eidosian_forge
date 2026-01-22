from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def compute_rho_estimate(self):
    x = self.work.x
    y = self.work.y
    z = self.work.z
    P = self.work.data.P
    q = self.work.data.q
    A = self.work.data.A
    pri_res = la.norm(A.dot(x) - z, np.inf)
    pri_res /= np.max([la.norm(A.dot(x), np.inf), la.norm(z, np.inf)]) + 1e-10
    dua_res = la.norm(P.dot(x) + q + A.T.dot(y), np.inf)
    dua_res /= np.max([la.norm(A.T.dot(y), np.inf), la.norm(P.dot(x), np.inf), la.norm(q, np.inf)]) + 1e-10
    new_rho = self.work.settings.rho * np.sqrt(pri_res / (dua_res + 1e-10))
    return min(max(new_rho, RHO_MIN), RHO_MAX)