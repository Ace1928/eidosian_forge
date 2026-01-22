from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def warm_start_x(self, x):
    """
        Warm start primal variable
        """
    self.work.settings.warm_start = True
    self.work.x = x
    self.work.x = self.work.scaling.Dinv.dot(self.work.x)
    self.work.z = self.work.data.A.dot(self.work.x)
    self.work.y = np.zeros(self.work.data.m)