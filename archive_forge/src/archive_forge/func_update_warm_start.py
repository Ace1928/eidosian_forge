from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_warm_start(self, warm_start_new):
    """
        Update warm_start parameter
        """
    if (warm_start_new is not True) & (warm_start_new is not False):
        raise ValueError('warm_start should be either True or False')
    self.work.settings.warm_start = warm_start_new