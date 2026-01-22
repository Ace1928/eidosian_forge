from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_lower_bound(self, l_new):
    """
        Update lower bound without requiring factorization
        """
    if self.work.clear_update_time == 1:
        self.work.clear_update_time = 0
        self.work.info.update_time = 0.0
    self.work.timer = time.time()
    self.work.data.l = l_new
    if self.work.settings.scaling:
        self.work.data.l = self.work.scaling.E.dot(self.work.data.l)
    if not np.greater_equal(self.work.data.u, self.work.data.l).all():
        raise ValueError('Lower bound must be lower than' + ' or equal to upper bound!')
    self.reset_info(self.work.info)
    self.update_rho_vec()
    self.work.info.update_time += time.time() - self.work.timer