from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def cold_start(self):
    """
        Cold start optimization variables to zero
        """
    self.work.x = np.zeros(self.work.data.n)
    self.work.z = np.zeros(self.work.data.m)
    self.work.y = np.zeros(self.work.data.m)