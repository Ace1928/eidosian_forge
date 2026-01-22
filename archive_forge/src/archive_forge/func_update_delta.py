from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_delta(self, delta_new):
    """
        Update delta parameter for polish
        """
    if delta_new <= 0:
        raise ValueError('delta must be positive')
    self.work.settings.delta = delta_new